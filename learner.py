import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import numpy as np
import tensorflow as tf
from tensorflow import keras
from buffer import *

#writer = tf.summary.FileWriter("log")
#writer = tf.summary.create_file_writer("logs")
log_dir="logs/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#config = tf.ConfigProto()
#num_threads = os.cpu_count()
#tf.config.threading.set_inter_op_parallelism_threads(num_threads)
#tf.config.threading.set_intra_op_parallelism_threads(num_threads)
#config.gpu_options.allow_growth = True
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
#train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
#test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)



class DeepQ:
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

    def build_model(self, input_dimension, output_dimension,
                    nodes_per_layer: int = 128,  # TODO difference between 128 vs 256
                    layer_count: int = 1,
                    gamma: float = 0.99,
                    learning_rate: float = 0.001):
        self.gamma = gamma
        self.model = self.build_model_function(input_dimension, output_dimension,
                                      nodes_per_layer, layer_count, learning_rate)
        self.target_model = self.build_model_function(input_dimension, output_dimension,
                                             nodes_per_layer, layer_count, learning_rate)
        self.update_target_model()

    def get_name(self):
        return self.name

    def update_target_model(self):
        logging.debug('DeepQ - updateTargetModel')
        self.target_model.set_weights(self.model.get_weights())

    def log(self):
        pass

    def get_next_action(self, state):
        logging.debug('DeepQ - getNextAction')
        action_values = self.model.predict_on_batch(np.atleast_2d(state))[0]
        # action_values = self.model.predict(np.atleast_2d(state))[0]
        return np.argmax(action_values)

        # TODO test without target
        # statePrimes = sample.nextStates
        # Get the action values for state primes NOTE TARGET MODEL
        # target_prime_action_values = self.targetModel.predict(statePrimes)
        # return target_prime_action_values

    def update(self, sample: ReplayBuffer):
        # TODO refactor
        #TODO combine model predections
        states = sample.states
        next_states = sample.next_states
        #action_values = self.model.predict_on_batch(np.concatenate((states, next_states), axis=0))
        #current_all_action_values, current_all_prime_action_values = np.split(action_values, 2)

        current_all_action_values = np.array(self.model.predict_on_batch(states)) # explictly make array due to TF eager
        current_all_prime_action_values = self.model.predict_on_batch(next_states)
        target_all_prime_action_values = self.target_model.predict_on_batch(next_states)

        #for idx in range(len(samples)):
        #for idx, (action, reward, is_done) in enumerate(zip(sample.actions, sample.rewards, sample.isDones)):
        idx = 0
        #for action, reward, is_done in zip(sample.actions, sample.rewards, sample.isDones):
        #for idx, (action, reward, is_done) in enumerate(zip(sample.actions, sample.rewards, sample.isDones)):
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

        self.model.fit(x=sample.states, y=current_all_action_values, batch_size=len(sample), epochs=1, verbose=0,
                       callbacks=[tensorboard_callback])
        losses = self.model.train_on_batch(x=states, y=current_all_action_values, reset_metrics=False)
        train_loss(losses)
        #losses = self.model.train_on_batch(x=states, y=current_all_action_values, reset_metrics=False)
        #tf.summary.scalar("loss", 0.5, step=losses)
        #summary = tf.summary(value=[tf.Summary.Value(tag="loss", simple_value=losses)])
        #writer.add_summary(summary)
        #3return losses
        return 4

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
        return DeepQ(name="Vanilla DeepQ", q_prime_function=DeepQFactory.vanilla_q_prime,
                     build_model_function=DeepQFactory.vanilla_build_model, *args, **kwargs)

    @staticmethod
    def create_double_deep_q(*args, **kwargs) -> DeepQ:
        """
        Reduce the overestimations by breaking up action seleciton and action evaluation
        """
        return DeepQ(name="Double DeepQ", q_prime_function=DeepQFactory.double_deepq_q_prime,
                     build_model_function=DeepQFactory.vanilla_build_model, *args, **kwargs)

    @staticmethod
    def create_clipped_double_deep_q(*args, **kwargs) -> DeepQ:
        """
        Reduce the overestimations by breaking up action seleciton and action evaluation
        """
        return DeepQ(name="Clipped Double DeepQ", q_prime_function=DeepQFactory.clipped_double_deep_q_q_prime,
                     build_model_function=DeepQFactory.vanilla_build_model, *args, **kwargs)

    @staticmethod
    def create_duel_deep_q(*args, **kwargs) -> DeepQ:
        return DeepQ(name="Duel DeepQ", q_prime_function=DeepQFactory.vanilla_q_prime,
                     build_model_function=DeepQFactory.dueling_build_model, *args, **kwargs)

    @staticmethod
    def create_double_duel_deep_q(*args, **kwargs) -> DeepQ:
        return DeepQ(name="Double Duel DeepQ", q_prime_function=DeepQFactory.double_deepq_q_prime,
                     build_model_function=DeepQFactory.dueling_build_model, *args, **kwargs)

    @staticmethod
    def create_clipped_double_duel_deep_q(*args, **kwargs) -> DeepQ:
        return DeepQ(name="Clipped Double Duel DeepQ",
                     q_prime_function=DeepQFactory.clipped_double_deep_q_q_prime,
                     build_model_function=DeepQFactory.dueling_build_model, *args, **kwargs)

    # Different Model Construction Methods.
    @staticmethod
    def vanilla_build_model(input_dimension, output_dimension, nodes_per_layer, hidden_layer_count, learning_rate):
        inputs = keras.Input(shape=(input_dimension,))
        hidden_layer = inputs
        for _ in range(hidden_layer_count):
            hidden_layer = keras.layers.Dense(nodes_per_layer, activation='relu')(hidden_layer)
            #hidden_layer = keras.layers.BatchNormalization()(hidden_layer)
        predictions = keras.layers.Dense(output_dimension, activation='linear')(hidden_layer)
        model = keras.Model(inputs=inputs, outputs=predictions)
        #model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate, decay=1e-08), loss='mse')
        model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss='mse')
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


def mean(array):
    return keras.backend.mean(array, axis=-1, keepdims=True)
