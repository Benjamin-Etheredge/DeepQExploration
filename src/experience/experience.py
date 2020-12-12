# TODO make dataclass
# https://docs.python.org/3/library/dataclasses.html
import numpy as np

class Experience:

    @staticmethod
    def training_items(experiences: list):
        actions = []
        rewards = []
        is_dones = []
        states = [[] for _ in range(4)]
        next_states = [[] for _ in range(4)]
        for item in experiences:
            frame_count = len(item.state)
            for frame_idx in range(frame_count):
                frame = item.state[frame_idx]
                states[frame_idx].append(frame)
            actions.append(item.action)

            for frame_idx in range(frame_count):
                frame = item.next_state[frame_idx]
                next_states[frame_idx].append(frame)
            rewards.append(item.reward)
            is_dones.append(item.is_done)

        actions = np.array(actions, dtype=np.uint8)
        rewards = np.array(rewards, dtype=np.float32)
        is_dones = np.array(is_dones, dtype=np.bool_)

        return states, actions, next_states, rewards, is_dones



    def __init__(self, state, action, next_state, reward, is_done):
        self._state = state
        self._action = action
        self._next_state = next_state
        self._reward = reward
        self._is_done = is_done

    @property
    def state(self):
        return self._state

    @property
    def action(self):
        return self._action

    @property
    def next_state(self):
        return self._next_state

    @property
    def reward(self):
        return self._reward

    @property
    def is_done(self):
        return self._is_done


'''
from dataclasses import dataclass


@dataclass
class DataExperience:
    state: list
    action: int
    next_state: list
    reward: float
    is_done: bool
'''
