# TODO note, total steps does not currenlty reflect the strating buffer prep
from unittest import TestCase
from multiprocessing import Process
import time

# TODO cleanup imports
import gym
from agent import Agent
from buffer import ReplayBuffer
from utils.utils import convert_atari_frame
from experience import Experience
from learners import DeepQFactory
import random
random.seed()
import numpy as np
np.random.seed(4)
import tensorflow as tf
tf.random.set_seed(4)

class TestAgent(TestCase):
    CartPole_V0 = 'CartPole-v0'
    CartPole_V1 = 'CartPole-v1'
    MountainCar_V0 = 'MountainCar-v0'
    Acrobot_V1 = 'Acrobot-v1'
    LunarLander_v2 = 'LunarLander-v2'

    ALL_ENVIRONMENTS = [
        (CartPole_V0, 400),
        (CartPole_V1, 600),
        (MountainCar_V0, 1000),
        (Acrobot_V1, 500),
        (LunarLander_v2, 1000)]

    ALL_LEARNERS_CREATORS = [
        DeepQFactory.create_vanilla_deep_q,
        DeepQFactory.create_double_deep_q,
        DeepQFactory.create_clipped_double_deep_q,
        DeepQFactory.create_duel_deep_q,
        DeepQFactory.create_double_duel_deep_q,
        DeepQFactory.create_clipped_double_duel_deep_q
    ]

    def test_play(self, environment, learner_creator, *args, **kwargs):
        score = self.play(environment, learner_creator(), *args, **kwargs)
        _, _, reward_threshold = self.get_env_info(environment)
        with self.subTest(f"{environment}_{learner_creator().name}"):
            self.assertGreaterEqual(score, reward_threshold)

    def play(self, name, learner,
              nodes_per_layer: int,
              layer_count,
              learning_rate: float,
              random_choice_min_rate: float,
              sample_size: int,
              verbose: float,
              max_episodes: int = 9999999,
              name_prefix="",
              experience_creator=Experience,
              buffer_creator=ReplayBuffer,
              data_func=None,
              window: int = 4,
              target_network_interval=None,
              start_length=200000,
              end_length=1000000,
              random_decay_end=4000000,  # Decay rate used for DQN in Rainbow.
              *args, **kwargs):

        # Seed random variables
        #np.random.seed(4)
        #random.seed(4)  # TODO May not be needed

        env_name, max_episode_steps, reward_threshold = self.get_env_info(name)
        env = gym.make(env_name)
        if verbose > 2:
            # https://github.com/openai/gym/wiki/FAQ
            env = gym.wrappers.Monitor(env, '.videos/' + str(time()) + '/')


        #self.assertEqual(name, env_name, "Gym did not return the correct information.")

        if len(env.observation_space.shape) == 1:
            feature_count = env.observation_space.shape[0]
        else:
            feature_count = np.array(env.observation_space.shape)
            #feature_count[2] = window
            print(f"features: {feature_count}")
            feature_count = (*feature_count[0:-1], window)
            #feature_count = np.swapaxes(feature_count,-1, 0)
            #print(f"features: {feature_count}")
            #3for dim in env.observation_space.shape:
                #feature_count *= dim
        action_count = env.action_space.n
        # Scale gamma to approach zero near max_episode_steps
        gamma = 0.99
        #gamma = float(np.power(0.0001, 1. / max_episode_steps))
        learner.build_model(input_dimension=feature_count, output_dimension=action_count,
                            nodes_per_layer=nodes_per_layer,
                            learning_rate=learning_rate,
                            layer_count=layer_count,
                            gamma=gamma, *args, **kwargs)

        # TODO account for possible extra space from scoring

        agent = Agent(
            learner=learner,
            sample_size=sample_size,
            replay_buffer=buffer_creator(max_length=end_length, start_length=start_length),
            environment=env,
            reward_threshold=reward_threshold,
            random_choice_decay_min=random_choice_min_rate,
            max_episode_steps=max_episode_steps,
            max_episodes=max_episodes,
            early_stopping=False,
            verbose=verbose,
            seed=4,
            experience_creator=experience_creator,
            observation_processor=data_func,
            window=window,
            target_network_interval=target_network_interval,
            random_decay_end=random_decay_end,
            name_prefix=name_prefix)
        score = agent.play(max_episodes * max_episode_steps, verbose=verbose)
        #score = agent.score_model(100, verbose=0)

        #self.assertGreaterEqual(score, reward_threshold)
        return score

    def parallel_play(self, environment, max_episodes, learner_creator, que=None):
        #with self.subTest(f"{environment}_{learner_creator().name}"):
            #print(f"{environment}_{learner_creator().name}")
        score = self.play(environment, max_episodes, learner_creator())
        _, _, reward_threshold = self.get_env_info(environment)
        #print(f"done {environment}_{learner_creator().name}")
        #que.put((f"{environment}_{learner_creator().name}", score, reward_threshold))
        que.put_nowait((f"{environment}_{learner_creator().name}", score, reward_threshold))
        #return(f"{environment}_{learner_creator().name}", score, reward_threshold)

    ATARI_ENVS = [
        "SpaceInvaders-v0"
    ]

    def test_watch_atari(self, *args, **kwargs):
        kwargs['verbose'] = 10
        return self.play_atari(*args, **kwargs)

    def test_atari(self, *args, **kwargs):
        kwargs['verbose'] = 1
        return self.play_atari(*args, **kwargs)

    def play_atari(self, environment_name, name_prefix="", *args, **kwargs):
        return self.test_play(
            environment=environment_name,
            learner_creator=DeepQFactory.create_atari_clipped_double_duel_deep_q,
            name_prefix=name_prefix,
            sample_size=32,
            layer_count=1,
            learning_rate=0.00025,
            target_network_interval=32000,  # Rainbow value
            random_choice_min_rate=0.01,
            nodes_per_layer=512,
            window=4,
            data_func=convert_atari_frame,
            conv_nodes=[32, 64, 64],
            kernel_size=[8, 4, 3],
            conv_stride=[4, 2, 1],
            *args, **kwargs)

    def test_SpaceInvaders_v0(self):
        self.test_atari("SpaceInvaders-v0")

    def test_watch_SpaceInvaders_v0(self):
        self.test_watch_atari("SpaceInvaders-v0")

    def test_SpaceInvaders_v4(self, *args, **kwargs):
        self.test_atari("SpaceInvaders-v4")

    def test_Pong(self, *args, **kwargs):
        self.test_atari("Pong-v4", *args, **kwargs)

    def test_watch_breakout(self):
        self.test_watch_atari('Breakout-v4')

    def test_Breakout(self, *args, **kwargs):
        self.test_atari("Breakout-v4", *args, **kwargs)

    def test_vanilla_Breakout(self):
        self.test_Breakout(name_prefix="vanilla_")

    def test_double_Breakout(self):
        self.test_atari("Breakout-v4", name_prefix="double_", double_deep_q=True)

    def test_duel_Breakout(self):
        self.test_atari("Breakout-v4", name_prefix="duel_", is_dueling=True)

    def test_vanilla(self):
        self.test_atari("SpaceInvaders-v0", name_prefix="vanilla_")

    def test_double(self):
        self.test_atari("SpaceInvaders-v0", name_prefix="double_", double_deep_q=True)

    def test_duel(self):
        self.test_atari("SpaceInvaders-v0", name_prefix="duel_", is_dueling=True)

    def test_double_duel(self):
        self.test_atari("SpaceInvaders-v0", name_prefix="double_duel_", double_deep_q=True, is_dueling=True)

    def test_clipped_double_duel(self):
        self.test_atari("SpaceInvaders-v0", name_prefix="clipped_double_duel_", double_deep_q=True, is_dueling=True, clipped_double_deep_q=True)

    def test_profile_Space(self):
        self.test_atari("SpaceInvaders-v0",
                        start_length=1000000,
                        end_length=1000000,
                        random_decay_end=20000,
                        max_episodes=300)

    def test_profile_Breakout(self):
        self.test_atari("Breakout-v4", 
                        start_length=100000,
                        end_length=100000,
                        random_decay_end=100000, 
                        max_episodes=250)

    def test_Breakout_v4(self):
        self.test_atari("Breakout-v4")
        #self.test_atari("Breakout-v0")

    def test_VideoPinball(self):
        self.test_atari("VideoPinball-v0")

    #conv_nodes = [32, 64, 64], kernel_size = [8, 4, 3], conv_stride = [4, 2, 1])

    def test_play_all(self):
        # TODO debug pool io.TextIOWrapper object issue
        #pool = Pool(max(1, int(os.cpu_count()/2)))
        threads = []
        que = Queue()
        count = 0
        #args = [(environment, max_episodes, learner_creator)
                #for environment, max_episodes in self.ALL_ENVIRONMENTS for learner_creator in self.ALL_LEARNERS_CREATORS]

        #results = pool.apply(self.parallel_play, args)
        #for test_name, score, score_goal in results:
            #with self.subTest(test_name):
                #self.assertGreaterEqual(score, score_goal)

        for environment, max_episodes in self.ALL_ENVIRONMENTS:
            for learner_creator in self.ALL_LEARNERS_CREATORS:
                #with self.subTest(f"{environment}_{learner_creator().name}"):
                    #self.test_play(environment, /max_episodes, learner_creator())
                #self.parallel_play(environment, max_episodes, learner_creator)
                thread = Process(target=self.parallel_play, args=(environment, max_episodes, learner_creator, que))
                thread.start()
                count += 1
                threads.append(thread)
        #for thread in threads:
            #thread.join()
        # TODO pull results as they finish
        while count > 0:
            if not que.empty():
                test_name, score, score_goal = que.get()
                with self.subTest(test_name):
                    self.assertGreaterEqual(score, score_goal)
                count -= 1
            time.sleep(5)

        #all_done = False
        #while not all_doneben/tensorflow-serving-devel-gpu:latest:
            #all_done = True
            #for t in que:
                #if t.is_alive():
                    #all_done = False
                    #time.sleep(1)

    def get_env_info(self, env_name):
        all_envs = gym.envs.registry.all()
        self.assertIn(env_name, [env.id for env in all_envs], "Game choice not in gym registry.")
        ids = [env_spec for env_spec in all_envs if env_spec.id == env_name]
        env_spec = ids[0]
        return env_spec.id, env_spec.max_episode_steps, env_spec.reward_threshold

    def test_CartPole_V0(self):
        self.test_play(
            name='CartPole-v0',
            max_episodes=100,
            learner=DeepQFactory.create_clipped_double_duel_deep_q(),
            nodes_per_layer=32,
            layer_count=2,
            learning_rate=0.0001,
            random_choice_min_rate=0.0)

    def test_CartPole_V1(self):
        self.test_play(
            name='CartPole-v1',
            max_episodes=600,
            learner=DeepQFactory.create_clipped_double_duel_deep_q(),
            nodes_per_layer=32,
            layer_count=2,
            learning_rate=0.0001,
            random_choice_min_rate=0.0)

    def test_LunarLander_v2(self):
        self.test_play(
            name='LunarLander-v2',
            max_episodes=1000,
            learner=DeepQFactory.create_clipped_double_duel_deep_q(),
            nodes_per_layer=128,
            layer_count=2,
            learning_rate=0.0001,
            random_choice_min_rate=0.0)

if __name__ == '__main__':
    import unittest
    #unittest.main()
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestAgent)
    #suite = unittest.TestSuite()
    #suite.addTest(TestAgent.test_SpaceInvaders_v0)
    #unittest.TextTestRunner().run(suite)
    suite = unittest.TestSuite()
    suite.addTest(TestAgent("test_profile_Space"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
