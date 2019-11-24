import numpy as np
import logging
import matplotlib.pyplot as plt


class Scores:

    def __init__(self, score_count: int = 200):
        self.numberOfRewardsToAverageOver = score_count
        self.lastAvarage = 0
        self.lastScores = np.full(score_count, 0)
        self.indexOfNextScore = 0
        self.allScores = []
        self.allAverages = []

    def append(self, reward):
        self.allScores.append(reward)
        self.lastScores[self.indexOfNextScore] = reward
        self.indexOfNextScore += 1
        self.indexOfNextScore %= self.numberOfRewardsToAverageOver

    def reset(self):
        self.lastAvarage = 0
        self.lastScores = np.zeros(self.numberOfRewardsToAverageOver)
        self.indexOfNextScore = 0
        self.allScores = []
        self.allAverages = []

    def averageReward(self):
        averageReward = np.mean(self.lastScores)
        self.allAverages.append(averageReward)
        return averageReward

    def numberOfGoodLandings(cls):
        return (cls.lastScores > 0).sum()

    def plotB(cls, game_name=None, learner_name=None):
        plt.plot(cls.allAverages)
        y_label = ""
        if game_name is not None:
            y_label += game_name + "\n"
        if learner_name is not None:
            y_label += learner_name + "\n"
        plt.ylabel(y_label + 'Average Total Reward per 200 episode')
        plt.xlabel('Episode')
        plt.show()

    def plotA(self, game_name=None, learner_name=None):
        plt.plot(self.allScores)
        y_label = ""
        if game_name is not None:
            y_label += game_name + "\n"
        if learner_name is not None:
            y_label += learner_name + "\n"
        plt.ylabel(y_label + 'Total Reward per episode')
        plt.xlabel('Episode')
        plt.show()
