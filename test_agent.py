from unittest import TestCase
import gym
import numpy as np
import threading
from multiprocessing import Process
from multiprocessing import Pool
import sys
import os


from agent import *
from learner import *
from buffer import *


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

    def test_play(self, environment, max_episodes, learner_creator, *args, **kwargs):
        score = self.play(environment, max_episodes, learner_creator(), *args, **kwargs)
        _, _, reward_threshold = self.get_env_info(environment)
        with self.subTest(f"{environment}_{learner_creator().name}"):
            self.assertGreaterEqual(score, reward_threshold)

    def play(self, name, max_episodes, learner,
              nodes_per_layer=128,
              layer_count=2,
              learning_rate=0.001,
              random_choice_min_rate=0.00,
              sample_size=64,
              verbose=0,
              experience_creator=Experience,
              data_func=None,
              window=4,
              target_network_interval=None, *args, **kwargs):

        # Seed random variables
        np.random.seed(4)
        random.seed(4)  # TODO May not be needed

        env_name, max_episode_steps, reward_threshold = self.get_env_info(name)
        env = gym.make(env_name)


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
        gamma = float(np.power(0.0001, 1. / max_episode_steps))
        learner.build_model(input_dimension=feature_count, output_dimension=action_count,
                            nodes_per_layer=nodes_per_layer,
                            learning_rate=learning_rate,
                            layer_count=layer_count,
                            gamma=gamma, *args, **kwargs)

        #start_length = min(int(max_episodes/10) * max_episode_steps, 500)
        start_length = min(int(max_episodes) * max_episode_steps, 200000)
        # TODO account for possible extra space from scoring
        max_possible_step_count = start_length * 5

        agent = Agent(
            learner=learner,
            scorer=Scores(10),
            sample_size=sample_size,
            replay_buffer=AtariBuffer(max_length=max_possible_step_count, start_length=start_length),
            environment=env,
            reward_threshold=reward_threshold,
            random_choice_decay_min=random_choice_min_rate,
            max_episode_steps=max_episode_steps,
            max_episodes=max_episodes,
            early_stopping=False,
            verbose=verbose,
            experience_creator=experience_creator,
            observation_processor=data_func,
            window=window,
            target_network_interval=target_network_interval)
        step_count = agent.play(4000 * max_episode_steps, verbose=0)
        score = agent.score_model(100, verbose=0)

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

    def test_SpaceInvaders_v0(self):
        self.test_play(
            environment='SpaceInvaders-v0',
            max_episodes=50000,
            learner_creator=DeepQFactory.create_atari_clipped_double_duel_deep_q,
            sample_size=32,
            verbose=1,
            experience_creator=AtariExperience,
            layer_count=1,
            learning_rate=0.00025,
            random_choice_min_rate=0.1,
            nodes_per_layer=512,
            window=4,
            target_network_interval=10000,
            data_func=AtariExperience.gray_scale,
            conv_layer_count=2, conv_nodes=32, conv_increase_factor=2, kernel_size=8, conv_stride=2)


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

