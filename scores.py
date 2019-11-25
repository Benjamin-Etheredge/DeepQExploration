import numpy as np
import logging
import matplotlib.pyplot as plt


class Scores:

    def __init__(self, score_count: int = 200):
        self.numberOfRewardsToAverageOver = score_count
        self.last_average = 0
        self.last_scores = np.full(score_count, 0)
        self.index_of_next_score = 0
        self.all_scores = []
        self.all_averages = []

    def append(self, reward):
        self.all_scores.append(reward)
        self.last_scores[self.index_of_next_score] = reward
        self.index_of_next_score += 1
        self.index_of_next_score %= self.numberOfRewardsToAverageOver

    def reset(self):
        self.last_average = 0
        self.last_scores = np.zeros(self.numberOfRewardsToAverageOver)
        self.index_of_next_score = 0
        self.all_scores = []
        self.all_averages = []

    def average_reward(self):
        average_reward = np.mean(self.last_scores)
        self.all_averages.append(average_reward)
        return average_reward

    def numberOfGoodLandings(cls):
        return (cls.last_scores > 0).sum()

    def plotB(cls, game_name=None, learner_name=None):
        plt.plot(cls.all_averages)
        y_label = ""
        if game_name is not None:
            y_label += game_name + "\n"
        if learner_name is not None:
            y_label += learner_name + "\n"
        plt.ylabel(y_label + 'Average Total Reward per 200 episode')
        plt.xlabel('Episode')
        plt.show()

    def plotA(self, game_name=None, learner_name=None):
        plt.plot(self.all_scores)
        y_label = ""
        if game_name is not None:
            y_label += game_name + "\n"
        if learner_name is not None:
            y_label += learner_name + "\n"
        plt.ylabel(y_label + 'Total Reward per episode')
        plt.xlabel('Episode')
        plt.show()
