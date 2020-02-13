import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
#import tensorflow.compat.v1.keras.backend as K
#dtype = 'float16'
#K.set_floatx(dtype)
#K.set_epsilon(1e-4)

tf.disable_eager_execution()  # disable eager for performance boost
tf.set_random_seed(4)
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from buffer import *
#import  tensorflow.compat.v2.k

##writer = tf.summary.FileWriter("log")
#writer = tf.summary.create_file_writer("logs")
#config = tf.ConfigProto()
# TODO investigate making tf dataset to get boost from eager
#config.gpu_options.allow_growth = True
tf_config=tf.ConfigProto()
tf_config.gpu_options.allow_growth=True
sess = tf.Session(config=tf_config)
#train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
#train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
#test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)


class DeepQ:
    #@profile
    def __init__(self,
                 name,
                 q_prime_function,
                 build_model_function):

        self.name = name
        self.q_prime_function = q_prime_function
        self.build_model_function = build_model_function

        self.gamma = None

        self.model = None
        self.target_model = None
        #log_dir = f"logs/{self.name}" + datetime.now().strftime("%Y%m%d-%H%M%S")
        #self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        #self.update_count = 0
        #self.summary_writer = tf.summary.create_file_writer(log_dir)

    #@profile
    def build_model(self, input_dimension, output_dimension,
                    nodes_per_layer: int = 128,  # TODO difference between 128 vs 256
                    layer_count: int = 1,
                    gamma: float = 0.99,
                    learning_rate: float = 0.001, *args, **kwargs):
        self.gamma = gamma
        self.model = self.build_model_function(input_dimension, output_dimension,
                                      nodes_per_layer, layer_count, learning_rate, *args, **kwargs)
        #tf.summary.
        #self.model.name = "Live_Network"
        self.target_model = self.build_model_function(input_dimension, output_dimension,
                                             nodes_per_layer, layer_count, learning_rate, *args, **kwargs)
        #self.target_model.name = "Target_Network"
        #self.tensorboard_callback.set_model(self.model)
        self.update_target_model()

    #@profile
    def get_name(self):
        return self.name

    #@profile
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    #@profile
    def log(self):
        pass

    #@profile
    def get_next_action(self, state):
        # TODO this is terrible.... refactor. I just want it to run right now
        if len(state.shape) > 1:
            state = state[np.newaxis, :, :, :]

        action_values = self.model.predict_on_batch(state)[0]
        return np.argmax(action_values)

        # TODO test without target
        # statePrimes = sample.nextStates
        # Get the action values for state primes NOTE TARGET MODEL
        # target_prime_action_values = self.targetModel.predict(statePrimes)
        # return target_prime_action_values

    #@profile
    def update(self, sample: ReplayBuffer):
        # TODO refactor
        #TODO combine model predections
        states = np.array(sample.states)
        #from PIL import Image
        #im = Image.fromarray(states[0, :, :, :3])
        #im.save("img.jpeg")
        #im = Image.fromarray(states[4, :, :, :3])
        #im.save("img2.jpeg")
        next_states = np.array(sample.next_states)
        #action_values = self.model.predict_on_batch(np.concatenate((states, next_states), axis=0))
        #current_all_action_values, current_all_prime_action_values = np.split(action_values, 2)

        current_all_action_values = self.model.predict_on_batch(states)  # TODO invistaigate explictly make array due to TF eager
        current_all_prime_action_values = self.model.predict_on_batch(next_states)
        target_all_prime_action_values = self.target_model.predict_on_batch(next_states)

        for idx, (action, reward, is_done) in enumerate(sample.training_items):
            q_prime = 0
            if not is_done:
                # TODO refactor
                q_prime = self.q_prime_function(target_all_prime_action_values[idx],
                                                current_all_prime_action_values[idx])
                # q_prime = self.qPrime(target_all_prime_action_values[idx], current_all_action_values[idx])

            actual_value = (self.gamma * q_prime) + reward
            current_all_action_values[idx][action] = actual_value
            #idx += 1

        # TODO refactor
        losses = self.model.train_on_batch(x=states, y=current_all_action_values)
        return losses


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
        return DeepQ(name="Atari_Clipped_Double_Duel_DeepQ",
                     q_prime_function=DeepQFactory.vanilla_q_prime,
                     #q_prime_function=DeepQFactory.clipped_double_deep_q_q_prime,
                     build_model_function=DeepQFactory.vanilla_conv_build_model, *args, **kwargs)

    #@static create_atari()

    # Different Model Construction Methods.
    #from tensorflow.compat.v1.keras import Input, Model
    #from tensorflow.compat.v1.keras.layers import Dense,
    @staticmethod
    def vanilla_build_model(input_dimension, output_dimension, nodes_per_layer, hidden_layer_count, learning_rate):
        #model = keras.models.Sequential()
        #model.add()
        inputs = keras.Input(shape=(input_dimension,))
        hidden_layer = inputs
        for _ in range(hidden_layer_count):
            hidden_layer = keras.layers.Dense(nodes_per_layer, activation='relu')(hidden_layer)
            #hidden_layer = keras.layers.BatchNormalization()(hidden_layer)
        predictions = keras.layers.Dense(output_dimension, activation='linear')(hidden_layer)
        model = keras.Model(inputs=inputs, outputs=predictions)
        #model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate, decay=1e-08), loss='mse')
        #model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss='mse')
        model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=tf.keras.losses.Huber())
        #keras.utils.plot_model(model, to_file=f"model.png")
        return model

    @staticmethod
    def dueling_build_model(input_dimension, output_dimension, nodes_per_layer, hidden_layer_count, learning_rate):
        # inputs = keras.Input(shape=(env.observation_space.shape[0],))
        inputs = keras.Input(shape=(input_dimension,))

        # Build Advantage layer
        advantage_hidden_layer = inputs
        for _ in range(hidden_layer_count):
            advantage_hidden_layer = keras.layers.Dense(nodes_per_layer, activation='relu')(advantage_hidden_layer)
        # predictions = keras.layers.Dense(env.action_space.n)(advantage_hidden_layer)
        predictions_advantage = keras.layers.Dense(output_dimension, activation='linear')(advantage_hidden_layer)

        # Build Value layer
        value_hidden_layer = inputs
        for _ in range(hidden_layer_count):
            value_hidden_layer = keras.layers.Dense(nodes_per_layer, activation='relu')(value_hidden_layer)
        predictions_value = keras.layers.Dense(1, activation='linear')(value_hidden_layer)

        # Combine layers
        advantage_average = keras.layers.Lambda(mean)(predictions_advantage)
        #advantage_average = keras.layers.AveragePooling1D()(predictions_advantage)
        #advantage_average = keras.layers.Average()(predictions_advantage)
        # advantageAverage = keras.layers.Average()([predictionsAdvantage, predictionsAdvantage])
        # advantageAverage = keras.backend.mean(predictionsAdvantage)
        # print(advantageAverage)
        # advantageAverage = keras.backend.constant(advantageAverage, shape=(outputDimension, 1))

        advantage = keras.layers.Subtract()([predictions_advantage, advantage_average])

        predictions = keras.layers.Add()([advantage, predictions_value])

        model = keras.Model(inputs=inputs, outputs=predictions)
        # model.compile(optimizer=keras.optimizers.Adam(lr=self.learningRate, decay=self.LRdecayRate),
        model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss='mse')
        #Emodel.compile(optimizer=keras.optimizers.RMSprop(lr=cls.learningRate), loss= keras.losses.huber_loss)
        #model.compile(optimizer=keras.optimizers.RMSprop(lr=learning_rate), loss='mse')
        #model.compile(optimizer=keras.optimizers.RMSprop(lr=learning_rate), loss=keras.losses.Huber())
        # model.compile(optimizer=keras.optimizers.RMSprop(lr=cls.learningRate),
                      #loss='mse')
        # metrics=['accuracy'])
        #keras.utils.plot_model(model, to_file=f"duel_model.png")
        return model

    # Different Model Construction Methods.
    @staticmethod
    def vanilla_conv_build_model(input_dimensions, output_dimension, nodes_per_layer, hidden_layer_count, learning_rate,
                                 conv_nodes, kernel_size, conv_stride):
        #model = keras.models.Sequential()
        #model.add()
        #input_dimensions = list(input_dimensions) + [1]
        #tprint(input_dimensions)
        #input_dimensions = input_dimensions[0], input_dimensions[1]/2, input_dimensions[2]/2
        input_dimensions = (int(round(input_dimensions[0]/2))), int(round((input_dimensions[1]/2))), input_dimensions[2]
        inputs = keras.Input(shape=tuple(input_dimensions))  # we'll be using the past 4 frames
        hidden_layer = keras.layers.Lambda(lambda x: x / 255.0)(inputs)
        #hidden_layer = inputs
        '''
        for conv_count, kernel, stride in zip(conv_nodes, kernel_size, conv_stride):
            hidden_layer = keras.layers.Conv2D(filters=conv_count,
                                               kernel_size=kernel,
                                               strides=stride,
                                               activation='relu', data_format='channels_last')(hidden_layer)
            #activation = 'relu', data_format = 'channels_first')(hidden_layer)
            #hidden_layer = keras.layers.MaxPool2D(pool_size=(pool_size,pool_size))(hidden_layer)
            #conv_nodes *= conv_increase_factor
            #pool_size = int(max(1, pool_size / 2))
            #kernel_size = int(max(3, kernel_size / 2))
        '''
        hidden_layer = keras.layers.Flatten()(hidden_layer)

        for _ in range(hidden_layer_count+1):
            hidden_layer = keras.layers.Dense(nodes_per_layer, activation='relu')(hidden_layer)
            #hidden_layer = keras.layers.BatchNormalization()(hidden_layer)

        predictions = keras.layers.Dense(output_dimension, activation='linear')(hidden_layer)
        model = keras.Model(inputs=inputs, outputs=predictions)
        #TODO switch back optimizers and huber
        #model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate, decay=1e-08), loss='mse')
        #model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss='mse')
        model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=tf.keras.losses.Huber())
        #keras.utils.plot_model(model, to_file=f"model.png")
        return model


def mean(array):
    return keras.backend.mean(array, axis=-1, keepdims=True)
