import time
from collections import deque
from datetime import datetime
from timeit import default_timer as timer
from guppy import hpy

import gym

from learner import *
from scores import *


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
# from tensorboard import summary
# kI
# tf.compat.v1.disable_eager_execution()  # disable eager for performance boost
# tf.compat.v1.disable_eager_execution()  # disable eager for performance boost
# 3tf.enable_resource_variables()
# tf.compat.v2.dis
# tf.python.framework_ops.disable_eager_execution() # disable eager for performance boost
# num_threads = os.cpu_count()
# tf.config.threading.set_inter_op_parallelism_threads(num_threads)
# tf.config.threading.set_intra_op_parallelism_threads(num_threads)
# tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# TODO process every 4th move
def dummy_process(data):
    return data


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
                 # decay_type: str = Agent.DECAY_TYPE_LINEAR,
                 early_stopping: bool = True,
                 verbose=0,
                 seed=None,
                 experience_creator=Experience,
                 observation_processor=dummy_process,
                 window=4,
                 target_network_interval=None):

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
        self.env.seed(self.seed())
        self.env.action_space.seed(self.seed())
        # This is needed to keep multiple game windows from opening up when scoring
        self.scoring_env = deepcopy(self.env)
        self.scoring_env.seed(self.seed())
        self.scoring_env.action_space.seed(self.seed())
        self.random_action_rate = 1.0
        self.scores = scorer
        self.verbose = verbose
        self.steps_per_game_scorer = Scores(100)
        self.early_stopping = early_stopping
        if verbose >= 1:
            log_dir = f"logs/agent_{learner.name}_" + datetime.now().strftime("%Y%m%d-%H%M%S")
            # self.tensorboard_writer = tf.summary.create_file_writer(log_dir)
            self.tensorboard_writer = tf.summary.FileWriter(log_dir)
            tensorboard = keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=0,
                batch_size=sample_size,
                write_graph=True,
                write_grads=True
            )
            tensorboard.set_model(self.learner.model)
            # self.tensorboard_writer.add_graph(self.learner.model)
            # self.tensorboard_writer.add_graph()
        # self.tensorboard_writer.set_as_default()

        # Easily Adjusted hyperparameters
        if reward_threshold is None:
            reward_threshold = sys.maxsize
        self.reward_stopping_threshold = reward_threshold
        self.max_episode_steps = max_episode_steps
        self.max_episodes = max_episodes
        self.on_policy_check_interval = min(max_episodes // 10, 1000)

        if target_network_interval is None:
            self.target_network_updating_interval = int(self.max_episode_steps * 0.5)
        else:
            self.target_network_updating_interval = target_network_interval
        self.sample_size = sample_size
        self.log_triggering_threshold = max_episode_steps * 10  # log every 20 max game lengths
        # self.randomChoiceDecayRate = randomChoiceDecayRate
        self.decay_type = decay_type
        if random_choice_decay_min == 0:
            random_choice_decay_min = 0.0000000000000001
        if self.decay_type == 'linear':
            self.randomChoiceDecayRate = float(
                (1.0 - random_choice_decay_min) / (self.max_episodes - (self.max_episodes * .9)))
        else:
            self.randomChoiceDecayRate = float(np.power(random_choice_decay_min, 1. / (self.max_episodes)))
        # self.randomChoiceDecayRate = float(np.power(self.max_episode_steps*300, (1./0.05)))
        self.randomChoiceMinRate = random_choice_decay_min
        self.iterations = 0

        # TQDM Status Monitors setup

        # tf.summary.record_if(verbose > 0) # TODO learn more about this function
        status_bars_disabled = verbose < 2

        meter_bar_format_elapsed = "{desc}: {n_fmt} [Elapsed: {elapsed}, {rate_fmt}]"
        meter_bar_format = "{desc}: {n_fmt} [{rate_fmt}]"
        tracker_fmt = "{desc}: {total_fmt}"
        running_average_fmt = tracker_fmt + "[Goal: " + str(reward_threshold) + "]"

    def seed(self):
        seed = self.np_random_state.randint(0, 9999)
        assert (seed >= 0)
        # return np.random.randint(0, 99999)  # seed env with controllable random generator

    def is_done_learning(self):
        average_reward = self.scores.average_reward()
        # with self.tensorboard_writer.as_default():
        # tf.summary.scalar("off_policy_average_reward", 0.5, self.iterations)
        # self.tensorboard_writer.flush()

        # self.iterations+=1
        variance_of_scores = self.scores.get_variance()
        return self.scores.get_variance() <= abs(0.01 * self.reward_stopping_threshold)
        # return average_reward >= self.reward_stopping_threshold

    # TODO figure out how to make verbose checking wrapper
    def verbose_1_check(self, *args, **kwargs):
        if self.verbose >= 1:
            tag, value, step = kwargs['name'], kwargs['data'], kwargs['step']
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.tensorboard_writer.add_summary(summary, step)
        # func(*args, **kwargs)

    def shouldSelectRandomAction(self):
        return random.uniform(0, 1) < self.random_action_rate

    def shouldUpdateLearner(self):
        return self.replay_buffer.is_ready()

    def shouldUpdateLearnerTargetModel(self, iteration):
        return iteration % self.target_network_updating_interval == 0

    # TODO why should this be a property?
    def shouldDecayRandomChoiceRate(self):
        return self.replay_buffer.is_ready()

    def getNextAction(self, state, random_choice_rate=None):
        if self.shouldSelectRandomAction():
            return self.env.action_space.sample()
        else:
            return self.learner.get_next_action(state)

    def decayRandomChoicePercentage(self):
        # TODO set decay operator
        if self.decay_type == 'linear':
            self.random_action_rate = max(self.randomChoiceMinRate,
                                          (self.random_action_rate - self.randomChoiceDecayRate))
        else:
            self.random_action_rate = max(self.randomChoiceMinRate,
                                          (self.randomChoiceDecayRate * self.random_action_rate))
        # self.randomChoicePercentage = minRate + (maxRate - minRate) * np.exp(-decayRate * iteration)

    def update_learner(self):
        sample_idxs, sample = self.replay_buffer.sample(self.sample_size)
        # npSample = convertSampleToNumpyForm(sample)
        # self.learner.update(npSample)
        loss = self.learner.update(sample)
        self.replay_buffer.update(sample_idxs, loss)

        return loss

    # TODO implement actual logger
    def should_log(self, iteration):
        return iteration % self.log_triggering_threshold == 0

    def log(self):
        # print("info - optimizaer {0}, loss {1}, dequeAmount: {2}".format(optimizer, loss, dequeAmount))
        # TODO paramertize optimizer
        self.learner.log()
        self.replay_buffer.log()

    def render_game(self):
        # self.scoring_env.seed(self.seed())
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

    def play(self, step_limit=float("inf"), verbose: int = 0):

        self.prepare_buffer()
        if verbose > 3:
            self.score_model(1, verbose=verbose)

        h = hpy()
        h.heap()
        game_count = 0
        total_steps = 0
        start_time = timer()
        iteration_time = start_time
        convergence_counter = 0
        variance_counter = 0
        while total_steps <= step_limit and self.max_episodes > game_count:
            # print("Start Iteration: {}".format(game_count))
            # self.verbose_1_check(tf.summary.scalar, "epsilon_rate_per_game", data=self.random_action_rate,
            # step=game_count)
            self.verbose_1_check(name="epsilon_rate_per_game", data=self.random_action_rate, step=game_count)
            self.verbose_1_check(name="buffer_size_in_experiences", data=len(self.replay_buffer), step=game_count)
            buffer_size_in_GBs = self.replay_buffer.size
            self.verbose_1_check(name="buffer_size_in_GBs", data=buffer_size_in_GBs, step=game_count)
            # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])

            # self.update_target_model()  # updating between games seems to perform significantly better than every C steps
            game_count += 1
            average_reward = self.scores.average_reward()
            variance_of_scores = self.scores.get_variance()

            if self.replay_buffer.is_ready() and variance_of_scores < np.abs(0.01 * self.reward_stopping_threshold):
                variance_counter += 1
                if variance_counter > 100:
                    pass
            else:
                variance_counter = 0

            if game_count % self.on_policy_check_interval == 0:
                # mini_score = self.score_model(1)
                mini_score = self.score_model(1, self.replay_buffer, verbose=verbose)
                self.verbose_1_check(name="intermediate_on_policy_score", data=mini_score, step=game_count)
                # if self.early_stopping and mini_score >= self.reward_stopping_threshold or np.isclose(mini_score, self.reward_stopping_threshold, rtol=0.1):
                if self.early_stopping and mini_score >= self.reward_stopping_threshold:
                    actual_score = self.score_model(100)
                    self.verbose_1_check(name="on_policy_score", data=actual_score, step=game_count)
                    # self.verbose_1_check(tf.summary.scalar, "on_policy_score", data=actual_score, step=game_count)
                    # actual_score = self.score_model(100, self.replay_buffer)
                    if actual_score >= self.reward_stopping_threshold * (np.abs(self.reward_stopping_threshold) * 0.1):
                        return total_steps

            # Start a new game
            # self.env.seed(self.seed())
            # TODO extract process to method
            step = self.observation_processor(self.env.reset())
            # step_buffer = np.moveaxis(np.array([step for _ in range(self.window + 1)]), 0, -1)
            step_buffer = deque([step for _ in range(self.window + 1)], self.window + 1)
            np_step_buffer = np.stack(step_buffer, axis=2)
            self.replay_buffer.prep(step)
            is_done = False
            total_reward = 0
            game_steps = 0
            # self.learner.update_target_model()
            game_start_time = time.time()
            while not is_done:
                # tf.summary.scalar("epsilon_rate_per_step", data=self.random_action_rate, step=total_steps)
                if verbose > 2:
                    self.env.render()
                action_choice = self.getNextAction(np_step_buffer[:, :, 1:])
                # self.verbose_1_check(tf.summary.histogram, "action", action_choice, step=total_steps)
                total_steps += 1
                game_steps += 1
                next_step, reward, is_done, _ = self.env.step(action_choice)
                next_step = self.observation_processor(next_step)
                # step_buffer = np.concatenate((np_step_buffer[:, :, 1:], step[:, :, np.newaxis]), axis=2)
                step_buffer.append(next_step)
                np_step_buffer = np.stack(step_buffer, axis=2)
                total_reward += reward
                # TODO add prioirity
                experience = self.experience_creator(np_step_buffer[:, :, :-1], action_choice, np_step_buffer[:, :, 1:],
                                                     reward, is_done)
                self.replay_buffer.append(experience)
                # step = next_step

                if self.replay_buffer.is_ready():
                    loss = self.update_learner()
                    # tf.summary.scalar("", data=loss, step=total_steps)
                    self.verbose_1_check(name="loss", data=loss, step=total_steps)
                    # self.verbose_1_check(tf.summary.scalar, data=loss, step=total_steps)
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
                        self.verbose_1_check(name="target_model_updates",
                                             data=int(total_steps / self.target_network_updating_interval),
                                             step=game_count)
                        self.update_target_model()

                if verbose > 2 and self.should_log(total_steps):
                    self.log_play(game_count, iteration_time, start_time, step_limit, total_steps, verbose)

            game_stop_time = time.time()
            elapsed_seconds = game_stop_time - game_start_time
            moves_per_second = game_steps / elapsed_seconds
            self.verbose_1_check(name="move_per_second_per_game", data=moves_per_second, step=game_count)
            self.verbose_1_check(name="off_policy_game_score", data=total_reward, step=game_count)
            self.scores.append(total_reward)
            self.steps_per_game_scorer.append(game_steps)
            self.verbose_1_check(name="steps_per_game", data=game_steps, step=game_count)
            self.decayRandomChoicePercentage()

        # self.plot()
        # self.score_model()
        assert total_steps > 0
        return total_steps

    def update_target_model(self):
        self.learner.update_target_model()

    def log_play(self, iteration, iteration_time, start_time, step_limit, total_steps, verbose):
        current_time = timer()
        iteration_time = current_time
        self.log()
        if verbose > 3:
            self.render_game()

    def load_model(self, file_name):
        # self.learner.load
        pass

    def save_model(self, file_name):
        pass

    # TODO use void learner to combine methods
    def play_game(self, buffer=None, verbose: int = 0):
        total_reward = 0
        done = False
        # self.scoring_env.seed(self.seed())
        step = self.observation_processor(self.scoring_env.reset())
        # step_buffer = deque([step for _ in range(self.window+1)], max_length=self.window+1)
        step_buffer = deque([step for _ in range(self.window + 1)], self.window + 1)
        self.replay_buffer.prep(step)
        # kjstep_buffer = np.moveaxis(np.array([step for _ in range(self.window+1)]), 0, -1)
        # step_buffer = deque([], maxlen=self.window)
        np_step_buffer = np.stack(step_buffer, axis=2)
        step_count = 0

        while not done:
            if verbose > 3:
                self.scoring_env.render()
            action_choice = self.getNextAction(np_step_buffer[:, :, 1:])
            # action_choice = self.learner.get_next_action(np_step_buffer[:, :, 1:])
            # TODO build better policy evaluator
            step, reward, done, _ = self.scoring_env.step(action_choice)
            step_count += 1
            step = self.observation_processor(step)
            step_buffer.append(step)
            # step_buffer = np.roll(step_buffer, 1, axis=2)
            # step_buffer[:, :, -1] = step
            # step_buffer[-1] = np.concatenate((step_buffer[:, :, 1:], step[:, :, np.newaxis]), axis=2)
            # step_buffer.append(step)
            if buffer is not None:
                experience = self.experience_creator(np_step_buffer[:, :, :-1], action_choice, np_step_buffer[:, :, 1:],
                                                     reward, done)
                self.replay_buffer.append(experience)
            total_reward += reward
        return total_reward

    def score_model(self, games=150, buffer=None, verbose: int = 0):
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
        scores = [self.play_game(buffer, verbose) for _ in range(games)]
        return np.mean(scores)

    def plot(self, game_name=None, learner_name=None):
        self.scores.plotA(game_name, learner_name)
        self.scores.plotB(game_name, learner_name)
