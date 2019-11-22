import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class DeepQ:
    def __init__(self, inputDiminsion, outputDiminsion,
                 nodesPerLayer=128,  # TODO difference between 128 vs 256
                 numLayers=2,
                 batchSize=32,
                 gamma=0.99,
                 # gamma=0.999,
                 learningRate=0.0005
                 # LRdecayRate=0.
                 ):

        self.model = self.buildModel(inputDiminsion, outputDiminsion, nodesPerLayer, numLayers, learningRate)
        self.targetModel = self.buildModel(inputDiminsion, outputDiminsion, nodesPerLayer, numLayers, learningRate)
        self.updateTargetModel()

        self.nodesPerLayer = nodesPerLayer
        self.numLayers = numLayers

        self.batchSize = batchSize
        self.gamma = gamma
        self.learningRate = learningRate
        # TODO evaluate decyaing LR self.LRdecayRate = LRdecayRate

    def updateTargetModel(self):
        logging.debug('DeepQ - updateTargetModel')
        self.targetModel.set_weights(self.model.get_weights())

    # TODO pull out to factory method
    @staticmethod
    def buildModel(inputDiminsion, outputDiminsion, nodesPerLayer, numLayers, learningRate):
        logging.debug('DeepQ - buildModel')
        # inputs = keras.Input(shape=(env.observation_space.shape[0],))
        inputs = keras.Input(shape=(inputDiminsion,))
        hiddenLayer = keras.layers.Dense(nodesPerLayer, activation='relu')(inputs)
        for _ in range(numLayers - 1):
            hiddenLayer = keras.layers.Dense(nodesPerLayer, activation='relu')(hiddenLayer)
        # predictions = keras.layers.Dense(env.action_space.n)(hiddenLayer)
        predictions = keras.layers.Dense(outputDiminsion)(hiddenLayer)
        model = keras.Model(inputs=inputs, outputs=predictions)
        # model.compile(optimizer=keras.optimizers.Adam(lr=self.learningRate, decay=self.LRdecayRate),
        model.compile(optimizer=keras.optimizers.Adam(lr=learningRate),
                      # model.compile(optimizer=keras.optimizers.RMSprop(lr=cls.learningRate),
                      loss='mse',
                      metrics=['accuracy'])
        return model

    def log(self):
        print("info - nodesPerLayer: {0}".format(self.nodesPerLayer))
        print("info - numLayers: {0}".format(self.numLayers))
        print("info - learning rate: {0}".format(self.learningRate))
        print("info - gamma: {0}".format(self.gamma))
        print("info - batchSize: {0}".format(self.batchSize))

    def getNextAction(self, state):
        logging.debug('DeepQ - getNextAction')
        actionValues = self.model.predict(np.atleast_2d(state))[0]
        return np.argmax(actionValues)

        # TODO test without target
        statePrimes = sample.nextStates
        # Get the action values for state primes NOTE TARGET MODEL
        target_prime_action_values = self.targetModel.predict(statePrimes)
        return target_prime_action_values

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
        current_all_action_values = self.model.predict(sample.states)
        current_all_prime_action_values = self.model.predict(sample.nextStates)
        target_all_prime_action_values = self.targetModel.predict(sample.nextStates)

        idx = 0
        for primeActionValue, action, reward, isDone in zip(target_all_prime_action_values, sample.actions, sample.rewards, sample.isDones):
            q_prime = 0
            if not isDone:
                # TODO refactor
                q_prime = self.qPrime(target_all_prime_action_values[idx], current_all_prime_action_values[idx])
                #q_prime = self.qPrime(target_all_prime_action_values[idx], current_all_action_values[idx])

            actual_value = (self.gamma * q_prime) + reward
            current_all_action_values[idx][action] = actual_value
            idx += 1

        # TODO refactor

        self.model.fit(x=sample.states, y=current_all_action_values, batch_size=self.batchSize, epochs=1, verbose=0)


class DoubleDeepQ(DeepQ):
    """
    Reduce the overestimations by breaking up action seleciton and action evaluation
    """

    def qPrime(self, prime_action_values, action_values):
        max_action = np.argmax(prime_action_values)

        return prime_action_values[np.argmax(action_values)]

# NOTE Extends DoubleDeepQ and NOT vanilla DeepQ
class DuelDeepQ(DoubleDeepQ):

    # TODO pull out to factory method
    @staticmethod
    def buildModel(inputDiminsion, outputDiminsion, nodesPerLayer, numLayers, learningRate):
        logging.debug('DuelQ - buildModel')

        # inputs = keras.Input(shape=(env.observation_space.shape[0],))
        inputs = keras.Input(shape=(inputDiminsion,))

        # Build Advantage layer
        hiddenLayer = keras.layers.Dense(nodesPerLayer, activation='relu')(inputs)
        for _ in range(numLayers - 1):
            hiddenLayer = keras.layers.Dense(nodesPerLayer, activation='relu')(hiddenLayer)
        # predictions = keras.layers.Dense(env.action_space.n)(hiddenLayer)
        predictionsAdvantage = keras.layers.Dense(outputDiminsion)(hiddenLayer)

        # Build Value layer
        hiddenLayer = keras.layers.Dense(nodesPerLayer, activation='relu')(inputs)
        for _ in range(numLayers - 1):
            hiddenLayer = keras.layers.Dense(nodesPerLayer, activation='relu')(hiddenLayer)
        predictionsValue = keras.layers.Dense(1)(hiddenLayer)

        # Combine layers
        advantageAverage = keras.layers.Lambda(mean)(predictionsAdvantage)
        # advantageAverage = keras.layers.Average()([predictionsAdvantage, predictionsAdvantage])
        # advantageAverage = keras.backend.mean(predictionsAdvantage)
        print(advantageAverage)
        # advantageAverage = keras.backend.constant(advantageAverage, shape=(outputDiminsion, 1))

        advantage = keras.layers.Subtract()([predictionsAdvantage, advantageAverage])

        predictions = keras.layers.Add()([advantage, predictionsValue])

        model = keras.Model(inputs=inputs, outputs=predictions)
        # model.compile(optimizer=keras.optimizers.Adam(lr=self.learningRate, decay=self.LRdecayRate),
        model.compile(optimizer=keras.optimizers.Adam(lr=learningRate),
                      # model.compile(optimizer=keras.optimizers.RMSprop(lr=cls.learningRate),
                      loss='mse',
                      metrics=['accuracy'])
        keras.utils.plot_model(model, to_file='model.png')
        return model

# class SuperQ()

def mean(array):
    return keras.backend.mean(array, axis=1, keepdims=True)

