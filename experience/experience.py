class Experience:

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


from dataclasses import dataclass


@dataclass
class DataExperience:
    state: list
    action: int
    next_state: list
    reward: float
    is_done: bool
