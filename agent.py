import gym
import random
import logging
import matplotlib.pyplot as plt
from buffer import *
from copy import deepcopy

from scores import *

from timeit import default_timer as timer
from learner import *


class Agent:
    def __init__(self, learner: DeepQ,
                 replayBuffer: ReplayBuffer,
                 environment: gym.Env,
                 scorer: Scores = Scores(100),
                 reward_threshold: int = None,
                 max_episode_steps=None,
                 sample_size=128,
                 random_choice_decay_min: float = 0.01,
                 verbose=0):

        # TODO do not construct here. Dependency inject
        self.learner = learner
        self.replay_buffer = replayBuffer
        self.env = environment
        # This is needed to keep multiple game windows from opening up when scoring
        self.scoring_env = deepcopy(self.env)
        self.random_action_rate = 1.1
        self.scores = scorer
        self.verbose = verbose

        # Easily Adjusted hyperparameters
        self.reward_stopping_threshold = reward_threshold
        self.max_episode_steps = max_episode_steps
        self.target_network_updating_interval = self.max_episode_steps*1
        self.sample_size = sample_size
        #self.target_network_updating_interval = target_network_updating_interval
        self.log_triggering_threshold = max_episode_steps * 10  # log every 20 max game lengths
        #self.randomChoiceDecayRate = randomChoiceDecayRate
        if random_choice_decay_min == 0:
            random_choice_decay_min = 0.0000000000000001
        self.randomChoiceDecayRate = float(np.power(random_choice_decay_min, 1. / (self.max_episode_steps * 500)))
        #self.randomChoiceDecayRate = float(np.power(self.max_episode_steps*300, (1./0.05)))
        self.randomChoiceMinRate = random_choice_decay_min

    def is_done_learning(self):
        logging.debug('isDoneLearning')
        return self.scores.averageReward() >= self.reward_stopping_threshold

    def shouldSelectRandomAction(self):
        logging.debug('shouldSelectRandomAction')
        return random.uniform(0, 1) < self.random_action_rate

    def shouldUpdateLearner(self):
        logging.debug('shouldUpdateLearner')
        return self.replay_buffer.isReady()

    def shouldUpdateLearnerTargetModel(self, iteration):
        return iteration % self.target_network_updating_interval == 0

    # TODO why should this be a property?
    def shouldDecayRandomChoiceRate(self):
        logging.debug('shouldDecayRandomChoiceRate')
        return self.replay_buffer.isReady()

    def getNextAction(self, state, random_choice_rate=None):
        logging.debug('getNextAction')
        if self.shouldSelectRandomAction():
            logging.debug('selecting randomly')
            return self.env.action_space.sample()
        else:
            logging.debug('selecting non-randomly')
            return self.learner.getNextAction(state)

    def decayRandomChoicePercentage(self):
        logging.debug('decayRandomChoice')
        self.random_action_rate = max(self.randomChoiceMinRate,
                                      (self.randomChoiceDecayRate * self.random_action_rate))
        # self.randomChoicePercentage = minRate + (maxRate - minRate) * np.exp(-decayRate * iteration)

    def updateLearner(self):
        logging.debug('updateLearner')
        sample = self.replay_buffer.sample(self.sample_size)
        # npSample = convertSampleToNumpyForm(sample)
        # self.learner.update(npSample)
        self.learner.update(sample)

    # TODO implement actual logger
    def shouldLog(self, iteration):
        return iteration % self.log_triggering_threshold == 0

    def log(self):
        print(f"info - Game: {self.env.unwrapped.spec.id}")
        print("info - numberOfExperiences: {0}".format(len(self.replay_buffer)))
        print("info - randomRate: {0}".format(self.random_action_rate))
        print(f"info - Reward Target: {self.reward_stopping_threshold}")
        print("info - averageReward: {0}".format(self.scores.averageReward()))
        print("info - randomDecay: {0}".format(self.randomChoiceDecayRate))
        print("info - sampleSize: {0}".format(self.sample_size))
        print("info - targetNetworkThreshold: {0}".format(self.target_network_updating_interval))
        print(f"info - max_episode_steps: {self.max_episode_steps}")
        # print("info - optimizaer {0}, loss {1}, dequeAmount: {2}".format(optimizer, loss, dequeAmount))
        # TODO paramertize optimizer
        self.learner.log()
        self.replay_buffer.log()

    def render_game(self):
        step = self.scoring_env.reset()
        is_done = False
        while not is_done:
            self.scoring_env.render()
            action_choice = self.learner.getNextAction(step)
            _, _, is_done, _ = self.scoring_env.step(action_choice)

    def play(self, step_limit=float("inf"), verbose=0):
        iteration = 0
        total_steps = 0
        start_time = timer()
        iteration_time = start_time
        while not self.is_done_learning():
            # print("Start Iteration: {}".format(iteration))
            iteration += 1

            # Start a new game
            if iteration % 100 == 0:
                score = self.score_model(100)
                print(f"\nitermediate score: {score}\n")
                if score >= self.reward_stopping_threshold:
                    return total_steps

            step = self.env.reset()
            is_done = False
            total_reward = 0
            self.learner.update_target_model()
            while not is_done:
                if total_steps > step_limit:
                    return total_steps

                if verbose > 2:
                    self.env.render()
                action_choice = self.getNextAction(step)
                total_steps += 1
                next_step, reward, is_done, _ = self.env.step(action_choice)
                # TODO add prioirity
                experience = Experience(step, action_choice, next_step, reward, is_done)
                self.replay_buffer.append(experience)
                step = next_step

                if self.replay_buffer.isReady():
                    self.updateLearner()
                    self.decayRandomChoicePercentage()

                    #if self.shouldUpdateLearnerTargetModel(total_steps):
                        #self.learner.update_target_model()

                if verbose > 0 and self.shouldLog(total_steps):
                    current_time = timer()
                    print(f"\nAt Iteration: {iteration}")
                    print(f"Step Limit: {step_limit}")
                    print(f"At step: {total_steps}")
                    print(f"Iteration took: {round(current_time - iteration_time, 2)}s")
                    print(f"Total Time: {round(current_time - start_time, 2)}s")
                    iteration_time = current_time
                    self.log()
                    if verbose > 1:
                        self.render_game()

                total_reward += reward

            self.scores.append(total_reward)

        #self.plot()

        #self.score_model()
        return total_steps

    def load_model(self, file_name):
        #self.learner.load
        pass

    def save_model(self, file_name):
        pass


    def score_model(self, games=150, verbose=0):
        scores = Scores(score_count=150)

        for _ in range(games):
            total_reward = 0
            done = False
            step = self.scoring_env.reset()
            while not done:
                if verbose > 0:
                    self.scoring_env.render()
                action_choice = self.learner.getNextAction(step)
                step, reward, done, _ = self.scoring_env.step(action_choice)

                total_reward += reward

            scores.append(total_reward)

        return scores.averageReward()

    def plot(self, game_name=None, learner_name=None):
        self.scores.plotA(game_name, learner_name)
        self.scores.plotB(game_name, learner_name)
