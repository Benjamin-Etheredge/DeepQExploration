import os
os.environ["PYTHONHASHSEED"] = "0"
from tensorflow.keras.backend import argmax, mean, square

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Conv2D, Lambda, Flatten, concatenate, Subtract, Add
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import Huber
from .deep_q import DeepQ
from .custom_layers import BellmanLayer, QPrimeLayer, DoubleQPrimeLayer, DuelingCombiningLayer, ClippedDoubleQPrimeLayer
import numpy as np
np.random.seed(4)
import tensorflow as tf
tf.random.set_seed(4)
tf.compat.v1.set_random_seed(4)

tf.config.set_soft_device_placement(True) # Don't think this affect performance much
tf.debugging.set_log_device_placement(True)

#DEVICE = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'
# TODO explore manually placing action seleciton network on CPU. 
#      tensorflow doesn't seem to like it but why?

class DeepQFactory:
    # Different Q-prime computating functions
    @staticmethod
    def vanilla_q_prime(target_prime_action_values, prime_action_values) -> float:
        #return np.max(target_prime_action_values)
        return max(target_prime_action_values)

    @staticmethod
    def double_deepq_q_prime(target_prime_action_values, prime_action_values) -> float:
        max_action = np.argmax(prime_action_values)
        return target_prime_action_values[max_action]

    @staticmethod
    def clipped_double_deep_q_q_prime(target_prime_action_values, prime_action_values) -> float:
        max_action_1 = np.argmax(target_prime_action_values)
        max_action_2 = np.argmax(prime_action_values)
        return min(prime_action_values[max_action_1], target_prime_action_values[max_action_2])

    # Different Model Factory Methods
    @staticmethod
    def create_vanilla_deep_q(*args, **kwargs) -> DeepQ:
        return DeepQ(name="Vanilla_DeepQ", q_prime_function=DeepQFactory.vanilla_q_prime,
                     build_model_function=DeepQFactory.vanilla_build_model, *args, **kwargs)

    @staticmethod
    def create_double_deep_q(*args, **kwargs) -> DeepQ:
        """
        Reduce the overestimations by breaking up action seleciton and action evaluation
        """
        return DeepQ(name="Double_DeepQ", q_prime_function=DeepQFactory.double_deepq_q_prime,
                     build_model_function=DeepQFactory.vanilla_build_model, *args, **kwargs)

    @staticmethod
    def create_clipped_double_deep_q(*args, **kwargs) -> DeepQ:
        """
        Reduce the overestimations by breaking up action seleciton and action evaluation
        """
        return DeepQ(name="Clipped_Double_DeepQ", q_prime_function=DeepQFactory.clipped_double_deep_q_q_prime,
                     build_model_function=DeepQFactory.vanilla_build_model, *args, **kwargs)

    @staticmethod
    def create_duel_deep_q(*args, **kwargs) -> DeepQ:
        return DeepQ(name="Duel_DeepQ", q_prime_function=DeepQFactory.vanilla_q_prime,
                     build_model_function=DeepQFactory.dueling_build_model, *args, **kwargs)

    @staticmethod
    def create_double_duel_deep_q(*args, **kwargs) -> DeepQ:
        return DeepQ(name="Double_Duel_DeepQ", q_prime_function=DeepQFactory.double_deepq_q_prime,
                     build_model_function=DeepQFactory.dueling_build_model, *args, **kwargs)

    @staticmethod
    def create_clipped_double_duel_deep_q(*args, **kwargs) -> DeepQ:
        return DeepQ(name="Clipped_Double_Duel_DeepQ",
                     q_prime_function=DeepQFactory.clipped_double_deep_q_q_prime,
                     build_model_function=DeepQFactory.dueling_build_model, *args, **kwargs)

    @staticmethod
    def create_atari_new_vanilla(*args, **kwargs) -> DeepQ:
        return DeepQ(name="Atari_Testing_Vanilla",
                     #q_prime_function=DeepQFactory.vanilla_q_prime,
                     q_prime_function=DeepQFactory.vanilla_q_prime,
                     build_model_function=DeepQFactory.vanilla_conv_build_model_raw, *args, **kwargs)

    @staticmethod
    def create_deep_q(*args, **kwargs) -> DeepQ:
        return DeepQ(name="Atari_Testing_Arch",
                     #q_prime_function=DeepQFactory.vanilla_q_prime,
                     q_prime_function=None, # TODO not used anymore
                     build_model_function=DeepQFactory.vanilla_conv_build_model_raw, *args, **kwargs)


    @staticmethod
    def create_atari_clipped_double_duel_deep_q(*args, **kwargs) -> DeepQ:
        return DeepQ(name="Atari_Testing_Arch",
                     #q_prime_function=DeepQFactory.vanilla_q_prime,
                     q_prime_function=DeepQFactory.double_deepq_q_prime,
                     build_model_function=DeepQFactory.vanilla_conv_build_model_raw, *args, **kwargs)

    #@static create_atari()

    # Different Model Construction Methods.
    @staticmethod
    def vanilla_build_model(input_dimension, output_dimension, nodes_per_layer, hidden_layer_count, learning_rate):
        inputs = Input(shape=input_dimension)
        hidden_layer = inputs
        for _ in range(hidden_layer_count):
            hidden_layer = Dense(nodes_per_layer, activation='relu')(hidden_layer)
            # TODO explore batchnorm in RL.
            #hidden_layer = BatchNormalization()(hidden_layer)
        predictions = Dense(output_dimension, activation='linear')(hidden_layer)
        model = Model(inputs=inputs, outputs=predictions)
        # TODO do more testing on MSE vs Huber
        #model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate, epsilon=1.5e-4), loss=tf.keras.losses.Huber())
        model.compile(optimizer=Adam(lr=learning_rate), loss=Huber())
        return model

    @staticmethod
    def dueling_build_model(input_dimension, output_dimension, nodes_per_layer, hidden_layer_count, learning_rate):
        inputs = Input(shape=(input_dimension,))

        # Build Advantage layer
        advantage_hidden_layer = inputs
        for _ in range(hidden_layer_count):
            advantage_hidden_layer = Dense(nodes_per_layer, activation='relu')(advantage_hidden_layer)
        predictions_advantage = Dense(output_dimension, activation='linear')(advantage_hidden_layer)

        # Build Value layer
        value_hidden_layer = inputs
        for _ in range(hidden_layer_count):
            value_hidden_layer = Dense(nodes_per_layer, activation='relu')(value_hidden_layer)
        predictions_value = Dense(1, activation='linear')(value_hidden_layer)

        # Combine layers
        advantage_average = Lambda(mean)(predictions_advantage)

        advantage = Subtract()([predictions_advantage, advantage_average])

        predictions = Add()([advantage, predictions_value])

        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=Adam(lr=learning_rate), loss=Huber())
        return model

    # Different Model Construction Methods.
    @staticmethod
    def vanilla_conv_build_model_raw(
            input_dimensions, 
            output_dimension, 
            nodes_per_layer, 
            hidden_layer_count, 
            learning_rate,
            conv_nodes,
            kernel_size,
            conv_stride,
            clip_reward=True,
            activation_name='relu',
            window=4,
            double_deep_q=False,
            clipped_double_deep_q=False,
            is_dueling=False,
            intermediate_features_diminsions=None):

        # TODO find better way to creat model. Should be able to reach into netowrk and grap/copy layers
        #      this is basically a playground for building tensorflow models
        #input_dimensions = (int(input_dimensions[0])//2, int(input_dimensions[1])//2)
        input_dimensions = (84, 84) # TODO ... so bad
        #input_dimensions = (105, 80) # TODO ... so bad
        #TODO test setting batch size for speedup
        state_frames = [Input(shape=input_dimensions, name=f"state_frame_{idx}", dtype=tf.uint8, batch_size=32) for idx in range(window)]
        action_selector_frames = [Input(shape=input_dimensions, name=f"action_selection_frame_{idx}", dtype=tf.uint8, batch_size=1) for idx in range(window)]
        next_state_frames = [Input(shape=input_dimensions, name=f"next_state_frame_{i}", dtype=tf.uint8, batch_size=32) for i in range(window)]

        frame_stacker = Lambda(lambda x: tf.stack(x, axis=-1), name="stack_frames")
        states = tf.cast(frame_stacker(state_frames), dtype=tf.float32)
        next_states = tf.cast(frame_stacker(next_state_frames), dtype=tf.float32)
        action_selector_states = tf.cast(frame_stacker(action_selector_frames), dtype=tf.float32)

        action = Input(shape=(1,), batch_size=32, dtype=tf.uint8, name='action')
        is_done = Input(shape=(1,), batch_size=32, dtype=tf.bool, name='is_done')
        reward = Input(shape=(1,), batch_size=32, name='reward')
        adjusted_reward = Lambda(lambda x: K.clip(x, -1, 1), name="clip_reward")(reward) if clip_reward else reward

        layers = []
        normalize_frames = Lambda(lambda x: x / 255.0, name="normalize_frames", dtype=tf.float32)
        layers.append(normalize_frames)
        scaled_layer_states = normalize_frames(states)
        scaled_layer_next_states = normalize_frames(next_states)
        action_selector = normalize_frames(action_selector_states)

        #for idx, scaled_layer in enumerate([scaled_layer_states, scaled_layer_next_states]):

        hidden_layer1 = scaled_layer_states
        hidden_layer2 = scaled_layer_next_states
        if double_deep_q:
            double_deep_q_network = scaled_layer_next_states

        for idx, (conv_count, kernel, stride) in enumerate(zip(conv_nodes, kernel_size, conv_stride)):
            conv_layer = Conv2D(
                filters=conv_count,
                kernel_size=kernel,
                strides=stride,
                activation=activation_name,
                use_bias=False,
                data_format='channels_last')
            layers.append(conv_layer)
            hidden_layer1 = conv_layer(hidden_layer1)
            action_selector = conv_layer(action_selector)
            if double_deep_q:
                double_deep_q_network = conv_layer(double_deep_q_network)
                #if is_dueling:
                    #value_double_deep_q_network = conv_layer(value_double_deep_q_network)

            hidden_layer2 = Conv2D(
                filters=conv_count,
                kernel_size=kernel,
                strides=stride,
                activation=activation_name,
                use_bias=False,
                data_format='channels_last',
                name=f"target_conv2d_{idx}")(hidden_layer2)

        flatten = Flatten(name='flatten')
        layers.append(flatten)
        hidden_layer1 = flatten(hidden_layer1)
        hidden_layer2 = flatten(hidden_layer2)
        action_selector = flatten(action_selector)

        #######################################
        if intermediate_features_diminsions:
            intermediate_features_layer = Input(shape=intermediate_features_diminsions, name='semi_supervised_features')
            #(encoder, decoder, autoencoder) = ConvAutoencoder.build(
                #*input_dimensions, window, filters=(32, 64, 128, 256), latentDim=512)
            #(tencoder, tdecoder, tautoencoder) = ConvAutoencoder.build(
                #*input_dimensions, window, filters=(32, 64, 128, 256), latentDim=512)

            # TODO create layers as members to separate out model construction
            encoded_layer = flatten(encoder(states))
            tencoded_layer = flatten(tencoder(next_states))
            aencoded_layer = flatten(encoder(action_selector_states))
            #encoded_layer = flatten(encoder(state_frames))
            #tencoded_layer = flatten(tencoder(next_state_frames))
            #aencoded_layer = flatten(encoder(action_selector_frames))

            hidden_layer1 = tf.keras.layers.Concatenate(axis=1)([encoded_layer, hidden_layer1])
            hidden_layer2 = tf.keras.layers.Concatenate(axis=1)([tencoded_layer, hidden_layer2])
            action_selector = tf.keras.layers.Concatenate(axis=1)([aencoded_layer, action_selector])

        #######################################




        # TODO unroll and apply "layers" here

        if double_deep_q:
            double_deep_q_network = flatten(double_deep_q_network)
            #######################################
            if intermediate_features_diminsions:
                double_deep_q_network = tf.keras.layers.Concatenate(axis=1)([
                    tencoder(scaled_layer_states), double_deep_q_network
                ])
            #######################################
            if is_dueling:
                value_double_deep_q_network = double_deep_q_network

        if is_dueling:
            state_value_network = hidden_layer1
            value_action_selector = action_selector
            target_state_value_network = hidden_layer2

        # TODO create layers in list then apply outside to break up function
        for _ in range(hidden_layer_count):
            dense_layer = Dense(nodes_per_layer, activation=activation_name)
            layers.append(dense_layer)
            hidden_layer1 = dense_layer(hidden_layer1)
            action_selector = dense_layer(action_selector)

            target_dense_network = Dense(nodes_per_layer, activation=activation_name)
            hidden_layer2 = target_dense_network(hidden_layer2)

            if is_dueling:
                dense_value_layer = Dense(nodes_per_layer, activation=activation_name)
                state_value_network = dense_value_layer(state_value_network)
                target_state_value_network = Dense(nodes_per_layer, activation=activation_name)(target_state_value_network)
                value_action_selector = dense_value_layer(value_action_selector)
            if double_deep_q:
                double_deep_q_network = dense_layer(double_deep_q_network)
                if is_dueling:
                    value_double_deep_q_network = dense_value_layer(value_double_deep_q_network)

        if is_dueling:
            # Advantage Functionality
            advantage_values_layer = Dense(output_dimension, activation='linear', name=f'advantage')
            layers.append(advantage_values_layer)
            advantage_values = advantage_values_layer(hidden_layer1)
            action_selector_advantage = advantage_values_layer(action_selector)
            target_advantage_values = Dense(output_dimension, activation='linear', name=f'target_advantage')(hidden_layer2)

            # State Value Functionality
            state_value_layer = Dense(1, activation='linear', name="state_value")
            state_value = state_value_layer(state_value_network)
            target_state_value = Dense(1, activation='linear', name="target_state_value")(target_state_value_network)
            action_selector_value = state_value_layer(value_action_selector)

            action_values_layer = DuelingCombiningLayer(name="action_values")
            action_values = action_values_layer([advantage_values, state_value])
            action_selector = action_values_layer([action_selector_advantage, action_selector_value])
            target_action_values = DuelingCombiningLayer(name="target_actions_values")([target_advantage_values, target_state_value])
        else:
            action_values_layer = Dense(output_dimension, activation='linear', name=f'action_values')
            action_values = action_values_layer(hidden_layer1)
            action_selector = action_values_layer(action_selector)
            target_action_values = Dense(output_dimension, activation='linear', name=f'target_action_values')(hidden_layer2)

        #TODO switch back optimizers and huber

        action_selector_best_action = Lambda(lambda x: K.argmax(x, axis=1), name='best_action')(action_selector)
        action_selector = Model(inputs=action_selector_frames, outputs=action_selector_best_action)

        def custom_mse_loss(values, correct_values):
            def loss(_1, _2):
                return mean(K.square(correct_values - values), axis=-1)
            return loss

        # Using closure here
        def custom_huber_loss(y_pred, y_true):
            def huber_loss(_1, _2, clip_delta=1.0):
                error = y_true - y_pred
                cond  = K.abs(error) < clip_delta

                squared_loss = 0.5 * K.square(error)
                linear_loss  = clip_delta * (K.abs(error) - 0.5 * clip_delta)

                return tf.where(cond, squared_loss, linear_loss)
            return huber_loss

        target = Model(inputs=next_state_frames, outputs=target_action_values)

        # TODO
        if double_deep_q:
            if is_dueling:
                double_advantage_values = advantage_values_layer(double_deep_q_network)
                double_state_value = state_value_layer(value_double_deep_q_network)
                double_action_values = action_values_layer([double_advantage_values, double_state_value])
            else:
                double_action_values = action_values_layer(double_deep_q_network)
            #actual_target_action_value = Model(inputs=next_state_frames, outputs=double_deep_q_network)
            #q_prime_value = Q_Prime_Layer(None)([target_model_action_values, adjusted_reward, is_done])
            if clipped_double_deep_q:
                q_prime_value = ClippedDoubleQPrimeLayer()([double_action_values, target_action_values, adjusted_reward, is_done])
            else:
                q_prime_value = DoubleQPrimeLayer()([double_action_values, target_action_values, adjusted_reward, is_done])
        else:
            q_prime_value = QPrimeLayer()([target_action_values, adjusted_reward, is_done])

        test = BellmanLayer(name="actual_action_values")([action_values, action, q_prime_value])
        # The model doesn't NEED output but keras REQUIRES it to have some. Maybe can be fixed with subclassing.
        trainable = Model(inputs=[*state_frames, action, *next_state_frames, reward, is_done], outputs=action_values)

        trainable.compile(optimizer=Adam(lr=learning_rate), loss=custom_huber_loss(action_values, test))

        #autoencoder.compile(loss="mse", optimizer=Adam(lr=1e-5))

        #return trainable, target, action_selector, (encoder, decoder, autoencoder), (tencoder, tdecoder, tautoencoder)
        return trainable, target, action_selector
        # TODO what if we use the encoder as the conv layer of deep q?

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
class ConvAutoencoder:
    @staticmethod
    def build(width, height, depth, filters=(32, 64), latentDim=16):
        # initialize the input shape to be "channels last" along with
        # the channels dimension itself
        # channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1

        # define the input to the encoder
        inputs = Input(shape=inputShape)
        x = inputs
        #state_frames = [Input(shape=input_dimensions, name=f"next_state_frame_{i}", dtype=tf.uint8, batch_size=32) for i in range(window)]
        #x = Lambda(lambda x: tf.stack(x, axis=-1), name="stack_frames")(state_frames)
        #x = Lambda(lambda x: x / 255.0, name="normalize_frames", dtype=tf.float32)(x)
        # loop over the number of filters
        for f in filters:
            # apply a CONV => RELU => BN operation
            x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            #x = BatchNormalization(axis=chanDim)(x)
        # flatten the network and then construct our latent vector
        volumeSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latentDim)(x)
        # build the encoder model
        encoder = Model(inputs, latent, name="encoder")

        # ---------------------------------------------------------

        # start building the decoder model which will accept the
        # output of the encoder as its inputs
        latentInputs = Input(shape=(latentDim,))
        x = Dense(np.prod(volumeSize[1:]))(latentInputs)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
        # loop over our number of filters again, but this time in
        # reverse order
        for f in filters[::-1]:
            # apply a CONV_TRANSPOSE => RELU => BN operation
            x = Conv2DTranspose(f, (3, 3), strides=2,
                padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            #x = BatchNormalization(axis=chanDim)(x)

        # apply a single CONV_TRANSPOSE layer used to recover the
        # original depth of the image
        x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
        outputs = Activation("sigmoid")(x)
        outputs = Lambda(lambda x: x * 255.0, name="normalize_frames", dtype=tf.float32)(outputs)
        # build the decoder model
        decoder = Model(latentInputs, outputs, name="decoder")

        # ---------------------------------------------------------

        # our autoencoder is the encoder + decoder
        autoencoder = Model(inputs, decoder(encoder(inputs)),
            name="autoencoder")
        # return a 3-tuple of the encoder, decoder, and autoencoder
        return (encoder, decoder, autoencoder)