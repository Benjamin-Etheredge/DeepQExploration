from tensorflow.keras.backend import argmax, mean, square

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Conv2D, Lambda, Flatten, concatenate, Subtract, Add
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from .deep_q import DeepQ
from .custom_layers import MyLayer, Q_Prime_Layer
import tensorflow as tf
import numpy as np
np.random.seed(4)
import tensorflow as tf
tf.random.set_seed(4)
tf.compat.v1.set_random_seed(4)

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
    def create_atari_clipped_double_duel_deep_q(*args, **kwargs) -> DeepQ:
        return DeepQ(name="Atari_Testing_Arch",
                     #q_prime_function=DeepQFactory.vanilla_q_prime,
                     q_prime_function=DeepQFactory.double_deepq_q_prime,
                     build_model_function=DeepQFactory.vanilla_conv_build_model_raw, *args, **kwargs)

    #@static create_atari()

    # Different Model Construction Methods.
    @staticmethod
    def vanilla_build_model(input_dimension, output_dimension, nodes_per_layer, hidden_layer_count, learning_rate):
        inputs = Input(shape=(input_dimension,))
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

    def double_custom_loss(layer):
        def loss(y):
            pass

    # Different Model Construction Methods.
    @staticmethod
    def vanilla_conv_build_model_raw(input_dimensions, output_dimension, nodes_per_layer, hidden_layer_count, learning_rate,
                                 conv_nodes, kernel_size, conv_stride):

        input_dimensions = tuple(((int(round(input_dimensions[0]/2))), int(round((input_dimensions[1]/2)))))
        state_frames = [Input(shape=input_dimensions, name=f"state_frame_{idx}")  for idx in range(4)]
        states = Lambda(lambda x: tf.stack(x, axis=-1))(state_frames)

        next_state_frames = [Input(shape=input_dimensions, name=f"next_state_frame_{i}") for i in range(4)]
        next_states = Lambda(lambda x: tf.stack(x, axis=-1))(next_state_frames)

        action = Input(shape=(1,), dtype=tf.int32, name='action') 
        is_done = Input(shape=(1,), dtype=tf.bool, name='is_done') 
        reward = Input(shape=(1,), name='reward') 

        scaled_layer_states = Lambda(lambda x: x / 255.0)(states)
        scaled_layer_next_states = Lambda(lambda x: x / 255.0)(next_states)
        networks = []
        for idx, scaled_layer in enumerate([scaled_layer_states, scaled_layer_next_states]):
            hidden_layer = scaled_layer

            for conv_count, kernel, stride in zip(conv_nodes, kernel_size, conv_stride):
                hidden_layer = Conv2D(filters=conv_count,
                                                   kernel_size=kernel,
                                                   strides=stride,
                                                   activation='relu',
                                                   use_bias=False,
                                                   data_format='channels_last')(hidden_layer)

            hidden_layer = Flatten()(hidden_layer)
            for _ in range(hidden_layer_count):
                hidden_layer = Dense(nodes_per_layer, activation='relu')(hidden_layer)

            predictions = Dense(output_dimension, activation='linear', name=f'action_values_{idx}')(hidden_layer)
            networks.append(predictions)

        #TODO switch back optimizers and huber
        #model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate, epsilon=1.5e-4), loss=tf.keras.losses.Huber())

        model_action_values, target_model_action_values = networks

        #with tf.device('/cpu:0'):
        best_action = Lambda(lambda x: K.argmax(x, axis=1), name='best_action')(model_action_values)
        action_selector = Model(inputs=state_frames, outputs=best_action)

        def custom_loss(values, correct_values):
            def loss(y_true, y_pred):
                return mean(K.square(correct_values - values), axis=-1)
            return loss

        def custom_huber_loss(y_pred, y_true):
            def huber_loss(_1, _2, clip_delta=1.0):
                error = y_true - y_pred
                cond  = K.abs(error) < clip_delta

                squared_loss = 0.5 * K.square(error)
                linear_loss  = clip_delta * (K.abs(error) - 0.5 * clip_delta)

                return tf.where(cond, squared_loss, linear_loss)
            return huber_loss

        # TODO double Q
        #target_action_value = Model(inputs=states, outputs=best_action)

        model = Model(inputs=state_frames, outputs=model_action_values)

        target = Model(inputs=next_state_frames, outputs=target_model_action_values)
        # TODO
        q_prime_value = Q_Prime_Layer(None)([model_action_values, target_model_action_values, reward, is_done])
        test = MyLayer(output_dimension)([model_action_values, action, q_prime_value])
        trainable = Model(inputs=[*state_frames, action, *next_state_frames, reward, is_done], outputs=model_action_values)

        #trainable.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=custom_loss(model_action_values, test))
        trainable.compile(optimizer=Adam(lr=learning_rate), loss=custom_huber_loss(model_action_values, test))

        #model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss={'values_0': tf.keras.losses.Huber()})
        return model, target, action_selector, trainable
