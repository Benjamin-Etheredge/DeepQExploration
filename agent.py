import gym
import random
import logging
import matplotlib.pyplot as plt
from buffer import *

from scores import *


class Agent:
    def __init__(self, learner, replayBuffer, environmentName,
                 experienceMaxThreshold=100000,
                 experienceStartThreshold=50000,
                 rewardThreshold=220,  # TODO could be higher
                 sampleSize=32,
                 targetNetworkThreshold=2000,  # threshold for updating target network
                 logThreshold=5000,
                 randomChoiceDecayRate=0.99999,
                 randomChoiceMinRate=0.1
                 ):

        self.env = gym.make(environmentName)
        numberOfFeatures = self.env.observation_space.shape[0]
        numberOfActions = self.env.action_space.n

        self.learner = learner(numberOfFeatures, numberOfActions)

        self.replayBuffer = replayBuffer(experienceMaxThreshold, experienceStartThreshold)
        self.randomChoicePercentage = 1.
        self.scores = Scores()

        # Easily Adjusted hyperparameters
        self.rewardThreshold = rewardThreshold
        self.sampleSize = sampleSize
        self.targetNetworkThreshold = targetNetworkThreshold
        self.logThreshold = logThreshold
        self.randomChoiceDecayRate = randomChoiceDecayRate
        self.randomChoiceMinRate = randomChoiceMinRate

    def isDoneLearning(self):
        logging.debug('isDoneLearning')
        return self.scores.averageReward() >= self.rewardThreshold

    def shouldSelectRandomAction(self):
        logging.debug('shouldSelectRandomAction')
        return random.uniform(0, 1) < self.randomChoicePercentage

    def shouldUpdateLearner(self):
        logging.debug('shouldUpdateLearner')
        return self.replayBuffer.isReady()

    def shouldUpdateLearnerTargetModel(self, iteration):
        return iteration % self.targetNetworkThreshold == 0

    # TODO why should this be a property?
    def shouldDecayRandomChoiceRate(self):
        logging.debug('shouldDecayRandomChoiceRate')
        return self.replayBuffer.isReady()

    def getNextAction(self, state):
        logging.debug('getNextAction')
        if self.shouldSelectRandomAction():
            logging.debug('selecting randomly')
            return self.env.action_space.sample()
        else:
            logging.debug('selecting non-randomly')
            return self.learner.getNextAction(state)

    def decayRandomChoicePercentage(self):
        logging.debug('decayRandomChoice')
        self.randomChoicePercentage = max(self.randomChoiceMinRate,
                                          (self.randomChoiceDecayRate * self.randomChoicePercentage))
        # self.randomChoicePercentage = minRate + (maxRate - minRate) * np.exp(-decayRate * iteration)

    def updateLearner(self):
        logging.debug('updateLearner')
        sample = self.replayBuffer.sample(self.sampleSize)
        # npSample = convertSampleToNumpyForm(sample)
        # self.learner.update(npSample)
        self.learner.update(sample)

    # TODO implement actual logger
    def shouldLog(self, iteration):
        return iteration % self.logThreshold == 0

    def log(self):
        print("info - numberOfExperiences: {0}".format(len(self.replayBuffer)))
        print("info - randomRate: {0}".format(self.randomChoicePercentage))
        print("info - averageReward: {0}".format(self.scores.averageReward()))
        print("info - randomDecay: {0}".format(self.randomChoiceDecayRate))
        print("info - sampleSize: {0}".format(self.sampleSize))
        print("info - targetNetworkThreshold: {0}".format(self.targetNetworkThreshold))
        # print("info - optimizaer {0}, loss {1}, dequeAmount: {2}".format(optimizer, loss, dequeAmount))
        # TODO paramertize optimizer
        self.learner.log()
        self.replayBuffer.log()

    def play(self):
        iteration = 0
        totalSteps = 0
        while not self.isDoneLearning():
            # print("Start Iteration: {}".format(iteration))
            iteration += 1

            # Start a new game
            step = self.env.reset()
            isDone = False
            totalReward = 0
            while not isDone:
                actionToTake = self.getNextAction(step)
                totalSteps += 1
                nextStep, reward, isDone, _ = self.env.step(actionToTake)
                # TODO add prioirity
                experience = Experience(step, actionToTake, nextStep, reward, isDone)
                self.replayBuffer.append(experience)
                step = nextStep

                if self.replayBuffer.isReady():
                    self.updateLearner()
                    self.decayRandomChoicePercentage()

                    if self.shouldUpdateLearnerTargetModel(totalSteps):
                        self.learner.updateTargetModel()

                if self.shouldLog(totalSteps):
                    print("At Iteration: {0}".format(iteration))
                    print("At step: {0}".format(totalSteps))
                    self.log()

                totalReward += reward

            self.scores.append(totalReward)

        self.plot()

        self.finalScoring()

    def finalScoring(self):
        self.scores.reset()
        self.randomChoicePercentage = 0

        for _ in range(100):
            totalReward = 0
            done = False
            step = np.array(env.reset())
            while not done:
                actionToTake = self.getNextAction(step, mainModel)
                step, reward, done, _ = self.env.step(actionToTake)
                totalReward += reward

            self.scores.append(totalReward)

        print("info - Average reward {}".format(Scores.averageReward()))
    def plot(self):
        self.scores.plotA()
        self.scores.plotB()

