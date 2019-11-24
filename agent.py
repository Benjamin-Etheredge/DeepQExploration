import gym
import random
import logging
import matplotlib.pyplot as plt
from buffer import *

from scores import *

from timeit import default_timer as timer


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
        start_time = timer()
        iteration_time = start_time
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
                    current_time = timer()
                    print(f"At Iteration: {iteration}")
                    print(f"At step: {totalSteps}")
                    print(f"Iteration took: {round(current_time - iteration_time, 2)}s")
                    print(f"Iteration took: {current_time - iteration_time}s")
                    print(f"Total Time: {round(current_time - start_time, 2)}s")
                    iteration_time = current_time
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
            step = np.array(self.env.reset())
            while not done:
                self.env.render()
                actionToTake = self.getNextAction(step)
                step, reward, done, _ = self.env.step(actionToTake)

                totalReward += reward

            self.scores.append(totalReward)

        print("info - Average reward {}".format(self.scores.averageReward()))

    def plot(self):
        self.scores.plotA()
        self.scores.plotB()
