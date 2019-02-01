import numpy as np
import logging
import matplotlib.pyplot as plt


class Scores:
    numberOfRewardsToAverageOver = 150

    def __init__(self):
        self.lastAvarage = 0
        self.lastScores = np.full(self.numberOfRewardsToAverageOver, -200)
        self.indexOfNextScore = 0
        self.allScores = []
        self.allAverages = []

    def append(cls, reward):
        cls.allScores.append(reward)
        cls.lastScores[cls.indexOfNextScore] = reward
        cls.indexOfNextScore += 1
        cls.indexOfNextScore %= cls.numberOfRewardsToAverageOver

    def reset(cls):
        cls.lastAvarage = 0
        cls.lastScores = np.zeros(cls.numberOfRewardsToAverageOver)
        cls.indexOfNextScore = 0
        cls.allScores = []
        cls.allAverages = []

    def averageReward(cls):
        averageReward = np.mean(cls.lastScores)
        cls.allAverages.append(averageReward)
        return averageReward

    def numberOfGoodLandings(cls):
        return (cls.lastScores > 0).sum()

    def plotB(cls):
        plt.plot(cls.allAverages)
        plt.ylabel('Average Total Reward per 100 episode')
        plt.xlabel('Episode')
        plt.show()

    def plotA(cls):
        plt.plot(cls.allScores)
        plt.ylabel('Total Reward per episode')
        plt.xlabel('Episode')
        plt.show()
