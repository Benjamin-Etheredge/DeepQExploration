import unittest
from unittest import TestCase
import gym
import numpy as np
np.random.seed(4)
import random
random.seed(4)
import copy
import os
os.environ["PYTHONHASHSEED"] = "0"

class TestEnv(TestCase):
    def test_random_gym_env(self, env_name="SpaceInvaders-v4", seed=None):
        env1 = gym.make(env_name)
        _ = env1.reset()
        done = False
        actions_1 = []
        while not done:
            action = env1.action_space.sample()
            actions_1.append(action)
            _, _, done, _ = env1.step(action)

        env2 = gym.make(env_name)
        _ = env2.reset()

        done = False
        actions_2 = []
        while not done:
            action = env2.action_space.sample()
            actions_2.append(action)
            _, _, done, _ = env2.step(action)

        self.assertFalse(actions_1 == actions_2, msg=f"actions_1: {actions_1}\nactions_2: {actions_2}")

    def test_seed_gym_env(self, env_name="SpaceInvaders-v4", seed=4):
        env1 = gym.make(env_name)
        env1.seed(seed)
        env1.action_space.seed(seed)
        _ = env1.reset()
        done = False
        actions_1 = []
        while not done:
            action = env1.action_space.sample()
            actions_1.append(action)
            _, _, done, _ = env1.step(action)

        env2 = gym.make(env_name)
        env2.seed(seed)
        env2.action_space.seed(seed)
        _ = env2.reset()

        done = False
        actions_2 = []
        while not done:
            action = env2.action_space.sample()
            actions_2.append(action)
            _, _, done, _ = env2.step(action)

        self.assertTrue(actions_1 == actions_2, msg=f"actions_1: {actions_1}\nactions_2: {actions_2}")

    def test_copy_seed_gym_env(self, env_name="SpaceInvaders-v4", seed=4):
        env1 = gym.make(env_name)
        env2 = copy.deepcopy(env1)
        env1.seed(seed)
        env1.action_space.seed(seed)
        _ = env1.reset()
        done = False
        actions_1 = []
        while not done:
            action = env1.action_space.sample()
            actions_1.append(action)
            _, _, done, _ = env1.step(action)

        env2 = copy.deepcopy(env1)
        env2.seed(seed)
        env2.action_space.seed(seed)
        _ = env2.reset()

        done = False
        actions_2 = []
        while not done:
            action = env2.action_space.sample()
            actions_2.append(action)
            _, _, done, _ = env2.step(action)

        self.assertTrue(actions_1 == actions_2, msg=f"actions_1: {actions_1}\nactions_2: {actions_2}")



if __name__ == '__main__':
    unittest.main()
