# TODO test with starting with large window and reducing size
# TODO test with randomly removing items from deque instead of using a sliding window
# TODO add new q value network for randomly sampling q values to test convergence of predicted q values.
import time
from datetime import datetime
from timeit import default_timer as timer
import sys
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
np.random.seed(4)
import tensorflow as tf
tf.random.set_seed(4)
import random
random.seed(4)

from numpy import clip, stack, array, power

#from learners import *
from learners import DeepQ
from tensorflow_core.python.keras.api._v1 import keras
from copy import deepcopy
import gym
#from scores import *
from experience import Experience
from buffer import ReplayBuffer, VoidBuffer
from collections import deque

# Limit GPU Memory Allocation
# https://mc.ai/tensorflow-2-0-wanna-limit-gpu-memory/
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import tensorflow.compat.v1 as tf # this must go after due to FileWriter. TODO cleanup
tf.disable_eager_execution()

# TODO process every 4th move

class Agent:
    DECAY_TYPE_LINEAR = 'linear'

    def __init__(self, learner: DeepQ,
                 replay_buffer: ReplayBuffer,
                 environment: gym.Env,
                 max_episode_steps: int,
                 max_episodes=float("inf"),
                 #scorer: Scores = Scores(100),
                 reward_threshold: int = None,
                 sample_size=128,
                 random_choice_decay_min: float = 0.05,
                 decay_type: str = 'linear',
                 # decay_type: str = Agent.DECAY_TYPE_LINEAR,
                 early_stopping: bool = True,
                 verbose=0,
                 seed=4,
                 #seed=None,
                 experience_creator=Experience,
                 observation_processor=array,
                 window=4,
                 frame_skip=3,
                 target_network_interval=None,
                 random_decay_end=1000000,
                 name_prefix="",
                 random_starting_actions_max=10):

        # seeding agents individually to achieve reproducible results across parallel runs.
        if seed is None:
            seed = np.random.randint(0, 99999999)
        self.np_random_state = np.random.RandomState(seed)
        self.experience_creator = experience_creator
        self.observation_processor = observation_processor
        self.window = window

        self.learner = learner
        self.replay_buffer = replay_buffer
        self.env = environment
        self.env.frameskip = frame_skip
        #self.env.seed(self.seed())
        self.env.seed(seed)
        #self.env.action_space.seed(self.seed())
        self.env.action_space.seed(seed)
        # This is needed to keep multiple game windows from opening up when scoring
        self.scoring_env = deepcopy(self.env)
        self.scoring_env.seed(seed)
        self.scoring_env.action_space.seed(seed)
        self.random_action_rate = 1.0
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.random_starting_actions_max = random_starting_actions_max
        if verbose >= 1:
            env_name = self.env.unwrapped.spec.id

            log_dir = f"logs/{name_prefix}{env_name}_{learner.name}_" + datetime.now().strftime("%Y%m%d-%H%M%S")
            self.tensorboard_writer = tf.summary.FileWriter(log_dir)
            tensorboard = keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=0,
                batch_size=sample_size,
                write_graph=True,
                write_grads=True
            )
            tensorboard.set_model(self.learner.model)

        # Easily Adjusted hyperparameters
        if reward_threshold is None:
            reward_threshold = sys.maxsize
        self.reward_stopping_threshold = reward_threshold
        self.max_episode_steps = max_episode_steps
        self.max_episodes = max_episodes
        self.on_policy_check_interval = min(max_episodes // 10, 150)

        if target_network_interval is None:
            self.target_network_updating_interval = int(self.max_episode_steps * 0.5)
        else:
            self.target_network_updating_interval = target_network_interval
        self.sample_size = sample_size
        self.log_triggering_threshold = max_episode_steps * 10  # log every 20 max game lengths
        self.decay_type = decay_type
        if random_choice_decay_min == 0:
            random_choice_decay_min = 0.0000000000000001
        if self.decay_type == 'linear':
            self.randomChoiceDecayRate = float(
                (1.0 - random_choice_decay_min) / random_decay_end)
        else:
            self.randomChoiceDecayRate = float(power(random_choice_decay_min, 1. / self.max_episodes))
        self.randomChoiceMinRate = random_choice_decay_min
        self.iterations = 0
        self.update_interval = 4
        self.frame_skip = frame_skip  # TODO push to custom gym wrapper
        self.prepare_buffer()

    def seed(self):
        seed = self.np_random_state.randint(0, 9999)
        assert (seed >= 0)
        return seed

    # TODO figure out how to make verbose checking wrapper
    def tensorboard_log(self, *args, **kwargs):
        if self.verbose >= 1:
            tag, value, step = kwargs['name'], kwargs['data'], kwargs['step']
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.tensorboard_writer.add_summary(summary, step)

    def should_select_random_action(self, random_choice_rate):
        return np.random.uniform(0, 1) < random_choice_rate

    def should_update_learner(self):
        return self.replay_buffer.is_ready()

    def should_update_autoencoder(self, iteration):
        return iteration % (self.target_network_updating_interval*20) == 0

    def should_update_target_model(self, iteration):
        return iteration % self.target_network_updating_interval == 0

    # TODO why should this be a property?
    def should_decay_epsilon(self):
        return self.replay_buffer.is_ready()

    def get_next_action(self, state, random_choice_rate=None):
        if random_choice_rate is None:
            random_choice_rate = self.random_action_rate
        if self.should_select_random_action(random_choice_rate):
            return self.get_random_action()
        else:
            return self.learner.get_next_action(state)

    def get_random_action(self):
        return self.env.action_space.sample()

    def decay_epsilon(self):
        # TODO set decay operator
        if self.decay_type == 'linear':
            self.random_action_rate = max(self.randomChoiceMinRate,
                                          (self.random_action_rate - self.randomChoiceDecayRate))
        else:
            self.random_action_rate = max(self.randomChoiceMinRate,
                                          (self.randomChoiceDecayRate * self.random_action_rate))

    def update_learner(self):
        sample_idxs, sample = self.replay_buffer.sample(self.sample_size)
        loss, learner_info = self.learner.update(sample)
        self.replay_buffer.update(sample_idxs, loss)
        return loss, learner_info

    # TODO implement actual logger
    def should_log(self, iteration):
        return iteration % self.log_triggering_threshold == 0

    def log(self):
        self.learner.log()
        self.replay_buffer.log()

    def render_game(self):
        self.play_game(verbose=10)

    def make_move(self, action):
        pass

    def prepare_buffer(self):
        while not self.replay_buffer.is_ready():
            self.play_game(self.replay_buffer, random_rate=1.0)

    def play(self, step_limit=float("inf"), verbose: int = 1):

        best_on_policy_score = float("-inf")
        best_off_policy_score = float("-inf")
        game_count = 0
        total_steps = 0
        rolling_average_scores = deque([], maxlen=200)
        moving_average = 0
        while total_steps <= step_limit and self.max_episodes > game_count:

            '''
            if game_count % self.on_policy_check_interval == 0:
                # Use max instead of min to be closer to the other publications
                # on_policy_score = np.mean([self.play_game(random_rate=0.0) for _ in range(4)])
                on_policy_scores = [self.play_game(random_rate=0.0) for _ in range(4)]
                max_on_policy_score = max(on_policy_scores)
                median_on_policy_score = np.median(on_policy_scores)
                if best_on_policy_score < max_on_policy_score:
                    best_on_policy_score = max_on_policy_score
                    self.tensorboard_log(name="best_on_policy_score", data=best_on_policy_score, step=total_steps)
                self.tensorboard_log(name="median_on_policy_score", data=median_on_policy_score, step=total_steps)
                self.tensorboard_log(name="max_on_policy_score_per_frames", data=max_on_policy_score, step=total_steps)
            '''

            game_count += 1

            # TODO extract process to method
            step = self.observation_processor(self.env.reset())
            list_buffer = [step for _ in range(self.window+1)]
            self.replay_buffer.prep(step)  # TODO is prep needed?

            current_lives = self.env.env.ale.lives()
            self.tensorboard_log(name="lives", data=current_lives, step=total_steps)
            is_done = False
            is_terminal = True
            total_reward = 0
            old_reward = 0
            old_steps = 0
            game_steps = 0
            game_start_time = time.time()

            # TODO for environments that reach the step limit, must specially handle case as not terminal
            # e.g. reaching the step limit should not have Q Prime set equal to 0.
            while not is_done:
                if verbose > 3:
                    self.env.render()

                if is_terminal:
                    starting_step = np.random.randint(1, self.random_starting_actions_max)  #should I be dividing this?
                    for _ in range(starting_step):
                        # TODO should make random, but breakout has a STUPID mechanic
                        # step, _, done, _ = self.scoring_env.step(self.get_random_action())
                        step, _, done, _ = self.env.step(1)
                        step = self.observation_processor(step)
                        #list_buffer.append(step)
                        #list_buffer.pop(0)
                    is_terminal = False
                    list_buffer = [step for _ in range(self.window + 1)]

                action_choice = self.get_next_action(list_buffer[1:])
                # self.verbose_1_check(tf.summary.histogram, "action", action_choice, step=total_steps)
                next_step, reward, is_done, info = self.env.step(action_choice)
                if 'ale.lives' in info:
                    lives = info['ale.lives']
                    is_terminal = lives < current_lives
                    if is_terminal:
                        self.tensorboard_log(name="life_reward",
                                             data=total_reward-old_reward,
                                             step=total_steps)
                        self.tensorboard_log(name="life_steps",
                                             data=game_steps-old_steps,
                                             step=total_steps)
                        old_reward = total_reward
                        old_steps = game_steps
                        self.tensorboard_log(name="lives", data=lives, step=total_steps)
                    current_lives = lives

                next_step = self.observation_processor(next_step)
                list_buffer.append(next_step)
                list_buffer.pop(0)
                total_reward += reward
                # TODO add prioirity
                experience = self.experience_creator(state=list_buffer[:-1],
                                                     action=action_choice,
                                                     next_state=list_buffer[1:],
                                                     reward=reward,
                                                     is_done=is_terminal or is_done)
                self.replay_buffer.append(experience)

                if self.replay_buffer.is_ready():
                    if total_steps % self.update_interval == 0:
                        loss, learner_info = self.update_learner()
                        self.tensorboard_log(name="loss", data=loss, step=total_steps)

                    self.decay_epsilon()

                    if self.should_update_target_model(total_steps):
                        self.tensorboard_log(name="target_model_updates",
                                             data=total_steps // self.target_network_updating_interval,
                                             step=total_steps)
                        self.update_target_model()
                    if self.should_update_autoencoder(total_steps):
                        pass
                        #self.learner.update_autoencoder(self.replay_buffer.states)
                total_steps += 1
                game_steps += 1

            game_stop_time = time.time()
            elapsed_seconds = game_stop_time - game_start_time
            moves_per_second = game_steps / elapsed_seconds
            best_off_policy_score = max(best_off_policy_score, total_reward)
            if best_off_policy_score < total_reward:
                best_off_policy_score = total_reward
                self.tensorboard_log(name="best_off_policy_score_per_frames", data=best_off_policy_score, step=total_steps)
            rolling_average_scores.append(total_reward)
            rolling_average = np.mean(rolling_average_scores)
            self.tensorboard_log(name="move_per_second", data=moves_per_second, step=total_steps)
            self.tensorboard_log(name="best_off_policy_score", data=best_off_policy_score, step=total_steps)
            self.tensorboard_log(name="off_policy_score", data=total_reward, step=total_steps)
            self.tensorboard_log(name="steps_per_game", data=game_steps, step=game_count)
            moving_average -= moving_average / game_count
            moving_average += total_reward / game_count
            self.tensorboard_log(name="rolling_average", data=rolling_average, step=total_steps)
            self.tensorboard_log(name="moving_average", data=moving_average, step=total_steps)

            self.tensorboard_log(name="epsilon_rate", data=self.random_action_rate, step=total_steps)
            self.tensorboard_log(name="buffer_size_in_experiences", data=len(self.replay_buffer), step=game_count)
            self.tensorboard_log(name="total steps", data=total_steps, step=game_count)

        assert total_steps > 0
        return best_off_policy_score, rolling_average, total_steps

    def update_target_model(self):
        self.learner.update_target_model()

    def load_model(self, file_name):
        pass

    def save_model(self, file_name):
        pass

    def play_game(self,
                  buffer=VoidBuffer(),
                  random_rate=0.0,
                  verbose: int = 0):
        total_reward = 0
        #self.scoring_env.seed(self.seed())
        #self.env.action_space.seed(self.seed())
        step = self.observation_processor(self.scoring_env.reset())
        list_buffer = [step for _ in range(self.window + 1)]
        current_lives = self.scoring_env.env.ale.lives()
        step_count = 0

        done = False
        is_terminal = True
        while not done:
            if verbose > 3:
                self.scoring_env.render()

            if is_terminal:
                starting_step = np.random.randint(1, self.random_starting_actions_max)  # should I be dividing this?
                for _ in range(starting_step):
                    # TODO should make random, but breakout has a STUPID fucking mechanic
                    # step, _, done, _ = self.scoring_env.step(self.get_random_action())
                    step, _, done, _ = self.scoring_env.step(1)
                    step = self.observation_processor(step)
                    #list_buffer.append(step)  # TODO should i be letting the list_buffer see these? probably not
                    #list_buffer.pop(0)
                list_buffer = [step for _ in range(self.window + 1)]
                is_terminal = False  # maybe not needed

            # TODO convert step_buffer to longer form and make it my window....
            # TODO but it probably won't make a huge difference since the np.arrays take way more space
            action_choice = self.get_next_action(list_buffer[1:], random_rate)
            # TODO build better policy evaluator
            step, reward, done, info = self.scoring_env.step(action_choice)
            total_reward += reward
            step_count += 1
            step = self.observation_processor(step)
            list_buffer.append(step)
            list_buffer.pop(0)

            if 'ale.lives' in info:
                lives = info['ale.lives']
                is_terminal = lives < current_lives
                current_lives = lives

            experience = self.experience_creator(
                state=list_buffer[:-1],
                action=action_choice,
                next_state=list_buffer[1:],
                reward=reward,
                is_done=done or is_terminal)
            buffer.append(experience)
        return total_reward

    def score_model(self, games=150, verbose: int = 0):
        scores = [self.play_game(verbose=verbose) for _ in range(games)]
        # Using max to be similar to other publications
        return max(scores)
