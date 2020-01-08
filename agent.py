# TODO test with starting with large window and reducing size
# TODO test with randomly removing items from deque instead of using a sliding window
# TODO switch linear degradation to per frame instead of per game
import time
from collections import deque
from datetime import datetime
from timeit import default_timer as timer
from guppy import hpy
import objgraph
import tracemalloc
tracemalloc.start(10)
#import tracemalloc
#snapshot = tracemalloc.take_snapshot()
#display_top(snapshot)
from copy import deepcopy


import gym
import gc

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
                 observation_processor=np.array,
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
        #self.scores = scorer
        self.verbose = verbose
        #self.steps_per_game_scorer = Scores(100)
        self.early_stopping = early_stopping
        if verbose >= 1:
            env_name = self.env.unwrapped.spec.id

            log_dir = f"logs/{env_name}_{learner.name}_" + datetime.now().strftime("%Y%m%d-%H%M%S")
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
            self.randomChoiceDecayRate = float(np.power(random_choice_decay_min, 1. / self.max_episodes))
        self.randomChoiceMinRate = random_choice_decay_min
        self.iterations = 0

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

    def should_select_random_action(self):
        return random.uniform(0, 1) < self.random_action_rate

    def should_update_learner(self):
        return self.replay_buffer.is_ready()

    def should_update_target_model(self, iteration):
        return iteration % self.target_network_updating_interval == 0

    # TODO why should this be a property?
    def should_decay_epsilon(self):
        return self.replay_buffer.is_ready()

    def get_next_action(self, state, random_choice_rate=None):
        if self.should_select_random_action():
            return self.env.action_space.sample()
        else:
            return self.learner.get_next_action(state)

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
        loss = self.learner.update(sample)
        self.replay_buffer.update(sample_idxs, loss)
        return loss

    # TODO implement actual logger
    def should_log(self, iteration):
        return iteration % self.log_triggering_threshold == 0

    def log(self):
        # TODO paramertize optimizer
        self.learner.log()
        self.replay_buffer.log()

    def render_game(self):
        self.play_game(verbose=10)

    def make_move(self, action):
        pass

    def prepare_buffer(self):
        while not self.replay_buffer.is_ready():
            self.play_game(self.replay_buffer)

    def play(self, step_limit=float("inf"), verbose: int = 0):

        self.prepare_buffer()

        game_count = 0
        total_steps = 0
        start_time = timer()
        while total_steps <= step_limit and self.max_episodes > game_count:

            game_count += 1

            # Start a new game
            # self.env.seed(self.seed())
            # TODO extract process to method
            step = self.observation_processor(self.env.reset())
            step_buffer = deque([step for _ in range(self.window + 1)], self.window + 1)
            list_buffer = list(step_buffer)
            self.replay_buffer.prep(step)

            is_done = False
            total_reward = 0
            game_steps = 0
            game_start_time = time.time()

            while not is_done:
                if verbose > 2:
                    self.env.render()
                action_choice = self.get_next_action(np.stack(list_buffer[1:], axis=2))
                # self.verbose_1_check(tf.summary.histogram, "action", action_choice, step=total_steps)
                total_steps += 1
                game_steps += 1
                next_step, reward, is_done, _ = self.env.step(action_choice)
                next_step = self.observation_processor(next_step)
                step_buffer.append(next_step)
                list_buffer = list(step_buffer)
                total_reward += reward
                # TODO add prioirity
                experience = self.experience_creator(state=list_buffer[:-1],
                                                     action=action_choice,
                                                     next_state=list_buffer[1:],
                                                     reward=np.clip(reward, -1, 1),
                                                     is_done=is_done)
                self.replay_buffer.append(experience)

                if self.replay_buffer.is_ready():
                    loss = self.update_learner()
                    self.tensorboard_log(name="loss", data=loss, step=total_steps)

                    # self.decayRandomChoicePercentage()

                    if self.should_update_target_model(total_steps):
                        self.tensorboard_log(name="target_model_updates",
                                             data=int(total_steps / self.target_network_updating_interval),
                                             step=game_count)
                        self.update_target_model()

                #if verbose > 2 and self.should_log(total_steps):
                    #self.log_play(game_count, iteration_time, start_time, step_limit, total_steps, verbose)

            game_stop_time = time.time()
            elapsed_seconds = game_stop_time - game_start_time
            moves_per_second = game_steps / elapsed_seconds
            #print(moves_per_second)
            self.tensorboard_log(name="move_per_second_per_game", data=moves_per_second, step=game_count)
            self.tensorboard_log(name="off_policy_game_score_per_game", data=total_reward, step=game_count)
            self.tensorboard_log(name="off_policy_game_score_per_frames", data=total_reward, step=total_steps)
            #self.scores.append(total_reward)
            #self.steps_per_game_scorer.append(game_steps)
            self.tensorboard_log(name="steps_per_game", data=game_steps, step=game_count)
            self.decay_epsilon()
            self.tensorboard_log(name="epsilon_rate_per_game", data=self.random_action_rate, step=game_count)
            self.tensorboard_log(name="epsilon_rate_per_frame", data=self.random_action_rate, step=total_steps)
            self.tensorboard_log(name="buffer_size_in_experiences", data=len(self.replay_buffer), step=game_count)
            self.tensorboard_log(name="total steps", data=total_steps, step=game_count)
            #buffer_size_in_GBs = self.replay_buffer.size
            #self.verbose_1_check(name="buffer_size_in_GBs", data=buffer_size_in_GBs, step=game_count)
            #gc.collect()
            '''
            if game_count % 100 == 0 or game_count == 0 or game_count == 1:
                #print(gc.collect())
                objgraph.show_refs([self], filename="agent.png")
                objgraph.show_refs([self.replay_buffer], filename="replay_buffer.png")
                objgraph.show_refs([self.learner], filename="learner.png")
                h = hpy()
                print(h.heap())
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')

                print("[ Top 10 ]")
                for stat in top_stats[:10]:
                    print(stat)
            '''

        assert total_steps > 0
        return total_steps

    def update_target_model(self):
        self.learner.update_target_model()

    def load_model(self, file_name):
        pass

    def save_model(self, file_name):
        pass

    # TODO use void learner to combine methods
    # TODO switch to np.clip(x, -1, 1)
    def play_game(self, buffer=VoidBuffer(), verbose: int = 0):
        total_reward = 0
        self.scoring_env.seed(self.seed())
        step = self.observation_processor(self.scoring_env.reset())
        step_buffer = deque([step for _ in range(self.window + 1)], self.window + 1)
        self.replay_buffer.prep(step)
        list_buffer = list(step_buffer)
        step_count = 0

        done = False
        while not done:
            if verbose > 3:
                self.scoring_env.render()
            # TODO convert step_buffer to longer form and make it my window....
            # TODO but it probably won't make a huge difference since the np.arrays take way more space            action_choice = self.getNextAction(np.stack(list_buffer[1:], axis=2))
            action_choice = self.get_next_action(np.stack(list_buffer[1:], axis=2))
            # TODO build better policy evaluator
            step, reward, done, _ = self.scoring_env.step(action_choice)
            step_count += 1
            step = self.observation_processor(step)
            step_buffer.append(step)
            list_buffer = list(step_buffer)
            if buffer is not None:
                experience = self.experience_creator(state=list_buffer[:-1],
                                                     action=action_choice,
                                                     next_state=list_buffer[1:],
                                                     reward=np.clip(reward, -1, 1),
                                                     is_done=done)
                self.replay_buffer.append(experience)
            total_reward += reward
        return total_reward

    def score_model(self, games=150, buffer=None, verbose: int = 0):
        scores = [self.play_game(buffer, verbose) for _ in range(games)]
        return np.mean(scores)
