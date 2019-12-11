import gym
import random
import logging
import matplotlib.pyplot as plt
from buffer import *
from copy import deepcopy
from tqdm import tqdm

from scores import *

from timeit import default_timer as timer
from learner import *
import multiprocessing
from multiprocessing import Process, Queue


class Agent:
    DECAY_TYPE_LINEAR = 'linear'

    def __init__(self, learner: DeepQ,
                 replay_buffer: ReplayBuffer,
                 environment: gym.Env,
                 max_episode_steps: int,
                 max_episodes=float("inf"),
                 scorer: Scores = Scores(100),
                 reward_threshold: int = None,
                 sample_size=128,
                 random_choice_decay_min: float = 0.05,
                 decay_type: str = 'linear',
                 early_stopping: bool = True,
                 verbose=0):

        self.learner = learner
        self.replay_buffer = replay_buffer
        self.env = environment
        # This is needed to keep multiple game windows from opening up when scoring
        self.scoring_env = deepcopy(self.env)
        self.random_action_rate = 1.0
        self.scores = scorer
        self.verbose = verbose
        self.steps_per_game_scorer = Scores(100)
        self.early_stopping = early_stopping

        # Easily Adjusted hyperparameters
        self.reward_stopping_threshold = reward_threshold
        self.max_episode_steps = max_episode_steps
        self.max_episodes = max_episodes
        self.target_network_updating_interval = int(self.max_episode_steps * 0.5)
        self.sample_size = sample_size
        self.log_triggering_threshold = max_episode_steps * 10  # log every 20 max game lengths
        # self.randomChoiceDecayRate = randomChoiceDecayRate
        self.decay_type = decay_type
        if random_choice_decay_min == 0:
            random_choice_decay_min = 0.0000000000000001
        if self.decay_type == 'linear':
            self.randomChoiceDecayRate = float(
                (1.0 - random_choice_decay_min) / (self.max_episodes - (self.max_episodes * .1)))
        else:
            self.randomChoiceDecayRate = float(np.power(random_choice_decay_min, 1. / (self.max_episodes)))
        # self.randomChoiceDecayRate = float(np.power(self.max_episode_steps*300, (1./0.05)))
        self.randomChoiceMinRate = random_choice_decay_min

        logging.info(f"Game: {self.env.unwrapped.spec.id}")
        logging.info(f"Reward Target: {self.reward_stopping_threshold}")
        logging.info(f"randomDecay: {self.randomChoiceDecayRate}")
        logging.info(f"sampleSize: {self.sample_size}")
        logging.info(f"targetNetworkThreshold: {self.target_network_updating_interval}")
        logging.info(f"max_episode_steps: {self.max_episode_steps}")

        # TQDM Status Monitors setup
        status_bars_disabled = verbose == 0
        meter_bar_format_elapsed = "{desc}: {n_fmt} [Elapsed: {elapsed}, {rate_fmt}]"
        meter_bar_format = "{desc}: {n_fmt} [{rate_fmt}]"
        tracker_fmt = "{desc}: {total_fmt}"
        running_average_fmt = tracker_fmt + "[Goal: " + str(reward_threshold) + "]"
        self.tqdm_graphs = []
        self.step_meter = self.create_tdm(desc="Steps", unit="steps", disable=status_bars_disabled,
                                          bar_format=meter_bar_format_elapsed)
        self.game_meter = self.create_tdm(total=self.max_episodes, desc="Games", unit="games",
                                          disable=status_bars_disabled)
        self.model_update_counter = self.create_tdm(desc="Model Updates", unit="updates", disable=status_bars_disabled,
                                                    bar_format=meter_bar_format)
        self.target_update_meter = self.create_tdm(desc="Target Updates", unit="updates", disable=status_bars_disabled,
                                                   bar_format=meter_bar_format)
        self.random_monitor = self.create_tdm(total=self.random_action_rate, desc="Current Random Action Rate",
                                              disable=status_bars_disabled, bar_format=tracker_fmt)
        self.game_step_monitor = self.create_tdm(desc="Average Steps per game over past 100 games",
                                                 disable=status_bars_disabled, bar_format=tracker_fmt)
        self.on_policy_monitor = self.create_tdm(desc="On-Policy Evaluation Score", unit="evals",
                                                 disable=status_bars_disabled, bar_format=running_average_fmt)
        self.off_policy_monitor = self.create_tdm(desc="Off-Policy Evaluation Score", unit="evals",
                                                  disable=status_bars_disabled, bar_format=running_average_fmt)
        self.variance_monitor = self.create_tdm(desc="Running Variance", unit="evals", disable=status_bars_disabled,
                                                bar_format=tracker_fmt)
        self.variance_count_monitor = self.create_tdm(desc="Variance Counts", unit="evals",
                                                      disable=status_bars_disabled, bar_format=tracker_fmt)
        self.loss_monitor = self.create_tdm(desc="Running Loss", unit="evals", disable=status_bars_disabled,
                                            bar_format=tracker_fmt)
        self.loss_counter = self.create_tdm(desc="Loss Counts", unit="evals", disable=status_bars_disabled,
                                            bar_format=tracker_fmt)

    def __del__(self):
        for graph in self.tqdm_graphs:
            graph.close()

    def create_tdm(self, bar_format=None, total=1, initial=0, desc="", unit="", disable=False, ):
        if bar_format is None:  # TODO combine if statement using kwargs
            #self.tqdm_graphs.append(tqdm(total=total, initial=initial, desc=desc, unit=unit, disable=disable, ascii=True))
            self.tqdm_graphs.append(tqdm(total=total, initial=initial, desc=desc, unit=unit, disable=disable, ncols=70))
        else:
            self.tqdm_graphs.append(tqdm(total=total, initial=initial, desc=desc, unit=unit, disable=disable,
                                         bar_format=bar_format, ncols=70))
        return self.tqdm_graphs[-1]

    def is_done_learning(self):
        logging.debug('isDoneLearning')
        average_reward = self.scores.average_reward()
        self.off_policy_monitor.total = average_reward
        self.off_policy_monitor.update(0)
        variance_of_scores = self.scores.get_variance()
        self.variance_monitor.total = variance_of_scores
        self.variance_monitor.update(0)
        return self.scores.get_variance() <= abs(0.01 * self.reward_stopping_threshold)
        # return average_reward >= self.reward_stopping_threshold

    def shouldSelectRandomAction(self):
        logging.debug('shouldSelectRandomAction')
        return random.uniform(0, 1) < self.random_action_rate

    def shouldUpdateLearner(self):
        logging.debug('shouldUpdateLearner')
        return self.replay_buffer.is_ready()

    def shouldUpdateLearnerTargetModel(self, iteration):
        return iteration % self.target_network_updating_interval == 0

    # TODO why should this be a property?
    def shouldDecayRandomChoiceRate(self):
        logging.debug('shouldDecayRandomChoiceRate')
        return self.replay_buffer.is_ready()

    def getNextAction(self, state, random_choice_rate=None):
        logging.debug('getNextAction')
        if self.shouldSelectRandomAction():
            logging.debug('selecting randomly')
            return self.env.action_space.sample()
        else:
            logging.debug('selecting non-randomly')
            return self.learner.get_next_action(state)

    def decayRandomChoicePercentage(self):
        logging.debug('decayRandomChoice')
        # TODO set decay operator
        if self.decay_type == 'linear':
            self.random_action_rate = max(self.randomChoiceMinRate,
                                          (self.random_action_rate - self.randomChoiceDecayRate))
        else:
            self.random_action_rate = max(self.randomChoiceMinRate,
                                          (self.randomChoiceDecayRate * self.random_action_rate))
        # self.randomChoicePercentage = minRate + (maxRate - minRate) * np.exp(-decayRate * iteration)
        self.random_monitor.total = self.random_action_rate
        self.random_monitor.update(0)

    def update_learner(self):
        logging.debug('updateLearner')  # TODO build warpper logs
        sample_idxs, sample = self.replay_buffer.sample(self.sample_size)
        # npSample = convertSampleToNumpyForm(sample)
        # self.learner.update(npSample)
        loss = self.learner.update(sample)
        self.replay_buffer.update(sample_idxs, loss)

        self.model_update_counter.update(1)
        self.loss_monitor.total = loss
        self.loss_monitor.update(0)

        return loss

    # TODO implement actual logger
    def should_log(self, iteration):
        return iteration % self.log_triggering_threshold == 0

    def log(self):
        logging.info(f"numberOfExperiences: {len(self.replay_buffer)}")
        logging.info(f"randomRate: {self.random_action_rate}")
        logging.info(f"averageReward: {self.scores.average_reward()}")
        # print("info - optimizaer {0}, loss {1}, dequeAmount: {2}".format(optimizer, loss, dequeAmount))
        # TODO paramertize optimizer
        self.learner.log()
        self.replay_buffer.log()

    def render_game(self):
        step = self.scoring_env.reset()
        is_done = False
        while not is_done:
            self.scoring_env.render()
            action_choice = self.learner.get_next_action(step)
            _, _, is_done, _ = self.scoring_env.step(action_choice)

    def make_move(self, action):
        pass

    def prepare_buffer(self):
        while not self.replay_buffer.is_ready():
            self.play_game(self.replay_buffer)

    def play(self, step_limit=float("inf"), verbose=0):

        self.prepare_buffer()

        iteration = 0
        total_steps = 0
        start_time = timer()
        iteration_time = start_time
        convergence_counter = 0
        variance_counter = 0
        while total_steps <= step_limit and self.max_episodes > iteration:
            # print("Start Iteration: {}".format(iteration))
            self.update_target_model()  # updating between games seems to perform significantly better than every C steps
            iteration += 1
            self.game_meter.update(1)
            average_reward = self.scores.average_reward()
            self.off_policy_monitor.total = average_reward
            self.off_policy_monitor.update(0)
            variance_of_scores = self.scores.get_variance()

            self.variance_monitor.total = variance_of_scores
            self.variance_monitor.update(0)
            if self.replay_buffer.is_ready() and variance_of_scores < np.abs(0.01 * self.reward_stopping_threshold):
                variance_counter += 1
                if variance_counter > 100:
                    pass
            else:
                variance_counter = 0

            self.variance_count_monitor.total = convergence_counter
            self.variance_count_monitor.update(0)

            if iteration % 25 == 0:
                # mini_score = self.score_model(1)
                mini_score = self.score_model(1, self.replay_buffer)
                self.on_policy_monitor.total = mini_score
                self.on_policy_monitor.update(0)
                logging.info(f"\nitermediate score: {mini_score}\n")
                # if self.early_stopping and mini_score >= self.reward_stopping_threshold or np.isclose(mini_score, self.reward_stopping_threshold, rtol=0.1):
                if self.early_stopping and mini_score >= self.reward_stopping_threshold:
                    actual_score = self.score_model(200)
                    # actual_score = self.score_model(100, self.replay_buffer)
                    self.on_policy_monitor.total = actual_score
                    self.on_policy_monitor.update(0)
                    if actual_score >= self.reward_stopping_threshold:
                        return total_steps

            # Start a new game
            step = self.env.reset()
            is_done = False
            total_reward = 0
            game_steps = 0
            # self.learner.update_target_model()
            while not is_done:
                if verbose > 2:
                    self.env.render()
                action_choice = self.getNextAction(step)
                total_steps += 1
                game_steps += 1
                self.step_meter.update(1)
                next_step, reward, is_done, _ = self.env.step(action_choice)
                total_reward += reward
                # TODO add prioirity
                experience = Experience(step, action_choice, next_step, reward, is_done)
                self.replay_buffer.append(experience)
                step = next_step

                if self.replay_buffer.is_ready():
                    loss = self.update_learner()
                    '''
                    if loss < 0.05:
                        convergence_counter += 1
                        if convergence_counter > 100:
                            pass
                    else:
                        convergence_counter = 0
                    self.loss_counter.total = convergence_counter
                    self.loss_counter.update(0)
                    '''

                    # self.decayRandomChoicePercentage()

                    if self.shouldUpdateLearnerTargetModel(total_steps):
                        # self.update_target_model()
                        pass

                if verbose > 0 and self.should_log(total_steps):
                    self.log_play(iteration, iteration_time, start_time, step_limit, total_steps, verbose)

            self.scores.append(total_reward)
            self.steps_per_game_scorer.append(game_steps)
            self.game_step_monitor.total = self.steps_per_game_scorer.average_reward()
            self.game_step_monitor.update(0)
            self.decayRandomChoicePercentage()

        # self.plot()

        # self.score_model()
        assert total_steps > 0
        return total_steps

    def update_target_model(self):
        self.target_update_meter.update(1)
        self.learner.update_target_model()

    def log_play(self, iteration, iteration_time, start_time, step_limit, total_steps, verbose):
        current_time = timer()
        logging.info(f"\nAt Iteration: {iteration}")
        logging.info(f"Step Limit: {step_limit}")
        logging.info(f"At step: {total_steps}")
        logging.info(f"Iteration took: {round(current_time - iteration_time, 2)}s")
        logging.info(f"Total Time: {round(current_time - start_time, 2)}s")
        iteration_time = current_time
        self.log()
        if verbose > 1:
            self.render_game()

    def load_model(self, file_name):
        # self.learner.load
        pass

    def save_model(self, file_name):
        pass

    def play_game(self, buffer=None):
        total_reward = 0
        done = False
        step = self.scoring_env.reset()
        previous_step = step
        while not done:
            action_choice = self.learner.get_next_action(step)
            step, reward, done, _ = self.scoring_env.step(action_choice)
            if buffer is not None:
                experience = Experience(previous_step, action_choice, step, reward, done)
                self.replay_buffer.append(experience)
                previous_step = step
            total_reward += reward
        return total_reward

    def play_game_worker(self):
        pass

    def score_model(self, games=150, verbose=0):
        """
        scores = Scores(score_count=games)

        for _ in range(games):
            score = self.play_game()
            scores.append(score)
        return scores.average_reward()
        #return np.mean(pool.map(self._map_play_game, range(games)))
        """
        # from functools import partial
        # partial_func = partial(play_game_parallel, self.learner)
        # pool = multiprocessing.Pool(4)
        # params = zip([self.learner] * games, pool.map(deepcopy, [self.scoring_env] * games))
        # temp = pool.map(play_game_parallel, params)
        # return_array = []
        # procs = []
        # for _ in range(games):
        # reward = play_game_parallel(self.learner, deepcopy(self.scoring_env))
        # proc = multiprocessing.Process(target=play_game_parallel, args=(self.learner, deepcopy(self.scoring_env), return_array))
        # proc = multiprocessing.Process(target=do_nothing)
        # procs.append(proc)
        # proc.start()
        # proc.join()

        # total_reward = self.play_game()
        # scores.append(total_reward)
        # for proc in procs:
        # proc.join()
        scores = [self.play_game() for _ in range(games)]
        return np.mean(scores)

    def plot(self, game_name=None, learner_name=None):
        self.scores.plotA(game_name, learner_name)
        self.scores.plotB(game_name, learner_name)


def play_game_parallel(model, env, shared):
    total_reward = 0
    done = False
    step = env.reset()
    while not done:
        action_choice = model.get_next_action(step)
        step, reward, done, _ = env.step(action_choice)
        total_reward += reward
    shared.append(total_reward)


def do_nothing():
    pass
