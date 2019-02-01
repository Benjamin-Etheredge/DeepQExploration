import logging

import numpy as np
from tensorflow._api.v1 import keras


class DeepQ:
    def __init__(self,
                 inputDiminsion,
                 outputDiminsion,
                 nodesPerLayer=128,  # TODO difference between 128 vs 256
                 numLayers=2,
                 batchSize=32,
                 gamma=0.995,
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
        allPrimeActionValues = self.targetModel.predict(statePrimes)
        return allPrimeActionValues

    # TODO maybe make method passed in?
    def qPrime(self, expectedValues, _):
        logging.debug('DeepQ - qPrime')
        return np.max(expectedValues)

    def update(self, sample):
        # TODO refactor
        states = sample.states
        allActionValues = self.model.predict(states)
        allPrimeActionValues = self.targetModel.predict(sample.nextStates)

        idx = 0
        for primeActionValue, action, reward, isDone in zip(allPrimeActionValues, sample.actions, sample.rewards, sample.isDones):
            qprime = 0
            if not isDone:
                # TODO refactor
                qprime = self.qPrime(allPrimeActionValues[idx], allActionValues[idx])

            actualValue = (self.gamma * qprime) + reward
            allActionValues[idx][action] = actualValue
            idx += 1

        # TODO refactor

        self.model.fit(x=states, y=allActionValues, batch_size=self.batchSize, epochs=1, verbose=0)


class DoubleDeepQ(DeepQ):

    def qPrime(self, primeActionValues, actionValues):
        return primeActionValues[np.argmax(actionValues)]

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

