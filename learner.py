import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow import keras


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class DeepQ:
    name = "DeepQ"
    def __init__(self, input_dimension, output_dimension,
                 nodesPerLayer=128,  # TODO difference between 128 vs 256
                 numLayers=1,
                 batchSize=32,
                 gamma=0.99,
                 # learningRate=0.0005
                 learningRate=0.001
                 ):

        # Hyperparameters
        self.nodes_per_layer = nodesPerLayer
        self.number_of_layers = numLayers
        self.batch_size = batchSize
        self.gamma = gamma

        self.learning_rate = learningRate
        # TODO evaluate decyaing LR
        # self.LRdecayRate = LRdecayRate

        self.model = self.build_model(input_dimension, output_dimension,
                                      self.nodes_per_layer, self.number_of_layers, self.learning_rate)
        self.target_model = self.build_model(input_dimension, output_dimension,
                                             self.nodes_per_layer, self.number_of_layers, self.learning_rate)
        self.update_target_model()

    @staticmethod
    def get_name():
        return "DeepQ"

    def update_target_model(self):
        logging.debug('DeepQ - updateTargetModel')
        self.target_model.set_weights(self.model.get_weights())

    # TODO pull out to factory method
    @staticmethod
    def build_model(input_dimension, output_dimension, nodes_per_layer, hidden_layer_count, learning_rate):
        logging.debug('DeepQ - buildModel')
        # inputs = keras.Input(shape=(env.observation_space.shape[0],))
        inputs = keras.Input(shape=(input_dimension,))
        hiddenLayer = inputs
        for _ in range(hidden_layer_count):
            hiddenLayer = keras.layers.Dense(nodes_per_layer, activation='relu')(hiddenLayer)
        # predictions = keras.layers.Dense(env.action_space.n)(hiddenLayer)
        predictions = keras.layers.Dense(output_dimension, activation='linear')(hiddenLayer)
        model = keras.Model(inputs=inputs, outputs=predictions)
        # model.compile(optimizer=keras.optimizers.Adam(lr=self.learningRate, decay=self.LRdecayRate),
        # model.compile(optimizer=keras.optimizers.RMSprop(lr=cls.learningRate),
        #model.compile(optimizer=keras.optimizers.RMSprop(lr=learning_rate), loss=keras.losses.Huber())
        model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss='mse')
        # metrics = ['accuracy'])
        keras.utils.plot_model(model, to_file=f"model.png")
        return model

    def log(self):
        print(f"info - name: {self.get_name()}")
        print("info - nodesPerLayer: {0}".format(self.nodes_per_layer))
        print("info - numLayers: {0}".format(self.number_of_layers))
        print("info - learning rate: {0}".format(self.learning_rate))
        print("info - gamma: {0}".format(self.gamma))
        print("info - batchSize: {0}".format(self.batch_size))

    def getNextAction(self, state):
        logging.debug('DeepQ - getNextAction')
        action_values = self.model.predict_on_batch(np.atleast_2d(state))[0]
        # action_values = self.model.predict(np.atleast_2d(state))[0]
        return np.argmax(action_values)

        # TODO test without target
        # statePrimes = sample.nextStates
        # Get the action values for state primes NOTE TARGET MODEL
        # target_prime_action_values = self.targetModel.predict(statePrimes)
        # return target_prime_action_values

    # TODO maybe make method passed in?
    def qPrime(self, prime_action_values, action_values):
        """
        Return the Q prime for Deep Q
        :param prime_action_values: expected values of actions for state prime
        :param action_values:
        :return:
        """
        logging.debug('DeepQ - qPrime')
        return np.max(prime_action_values)

    def update(self, sample):
        # TODO refactor
        current_all_action_values = np.array(self.model.predict_on_batch(sample.states))
        current_all_prime_action_values = self.model.predict_on_batch(sample.nextStates)
        target_all_prime_action_values = self.target_model.predict_on_batch(sample.nextStates)

        idx = 0
        for action, reward, is_done in zip(sample.actions, sample.rewards, sample.isDones):
            q_prime = 0
            if not is_done:
                # TODO refactor
                q_prime = self.qPrime(target_all_prime_action_values[idx], current_all_prime_action_values[idx])
                # q_prime = self.qPrime(target_all_prime_action_values[idx], current_all_action_values[idx])

            actual_value = (self.gamma * q_prime) + reward
            current_all_action_values[idx][action] = actual_value
            idx += 1

        # TODO refactor

        # self.model.fit(x=sample.states, y=current_all_action_values, batch_size=self.batchSize, epochs=1, verbose=0)
        self.model.train_on_batch(x=sample.states, y=current_all_action_values)


class DoubleDeepQ(DeepQ):
    name = "DoubleDeepQ"
    """
    Reduce the overestimations by breaking up action seleciton and action evaluation
    """

    @staticmethod
    def get_name():
        return "DoubleDeepQ"

    def qPrime(self, target_prime_action_values, prime_action_values):
        max_action = np.argmax(target_prime_action_values)
        return prime_action_values[max_action]


# NOTE Extends DoubleDeepQ and NOT vanilla DeepQ
class DuelDeepQ(DoubleDeepQ):
    name = "DuelDeepQ"

    @staticmethod
    def get_name():
        return "DuelDeepQ"


    # TODO pull out to factory method
    @staticmethod
    def build_model(input_dimension, output_dimension, nodes_per_layer, hidden_layer_count, learning_rate):
        logging.debug('DuelQ - buildModel')

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
        advantageAverage = keras.layers.Lambda(mean)(predictions_advantage)
        # advantageAverage = keras.layers.Average()([predictionsAdvantage, predictionsAdvantage])
        # advantageAverage = keras.backend.mean(predictionsAdvantage)
        # print(advantageAverage)
        # advantageAverage = keras.backend.constant(advantageAverage, shape=(outputDimension, 1))

        advantage = keras.layers.Subtract()([predictions_advantage, advantageAverage])

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
        keras.utils.plot_model(model, to_file=f"duel_model.png")
        return model


# class SuperQ()

def mean(array):
    return keras.backend.mean(array, axis=1, keepdims=True)
