import collections
import random
import sys

import numpy as np


# TODO implement priority replay buffer

class Experience:
    MEMORY_SIZE = None

    def __init__(self, state, action, next_state, reward, is_done):
        # self.data = [state, action, next_state, reward, is_done]
        self._state = state
        # print(self._state.shape)
        # if len(self.__state.shape) > 1:
        # self.__state = self.__state.flatten()
        self._action = action
        self._next_state = next_state
        # if len(self.__nextState.shape) > 1:
        # self.__nextState = self.__nextState.flatten()
        self.__reward = reward
        self.__isDone = is_done
        if Experience.MEMORY_SIZE is None:
            # try:
            size_1 = self._state.size * self._state.itemsize
            # except:
            # size_1 = 0
            try:
                size_2 = self._next_state.size * self._state.itemsize
            except AttributeError:
                size_2 = 0

            Experience.MEMORY_SIZE = (size_1 + size_2 +
                                      sys.getsizeof(self._action) + sys.getsizeof(self.__reward) +
                                      sys.getsizeof(self.__isDone)) / 1024. / 1024. / 1024.

    @classmethod
    def size(cls):
        return cls.MEMORY_SIZE

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
        return self.__reward

    @property
    def isDone(self):
        return self.__isDone


class AtariExperience(Experience):

    def __init__(self, state, action, next_state, reward, is_done):
        Experience.__init__(self, state, action, next_state[:, :, -1], reward, is_done)

    @property
    def next_state(self):
        return np.dstack((self._state[:, :, 1:], self._next_state))

    @staticmethod
    def gray_scale(img):
        return np.mean(img[::2, ::2], axis=2).astype(np.uint8)  # TODO reduce 3 -> 2


class ReplayBuffer:

    def __init__(self,
                 max_length: int = 100000,
                 start_length: int = None,
                 buffer: list = None):
        """
        Constructor for Default replay buffer
        :param max_length:
        :param start_length:
        :param buffer:
        """

        self.max_length = max_length

        if start_length is None:
            start_length = max_length
        self.start_length = start_length

        if buffer is None:
            # TODO deque may be slow for sampling
            buffer = collections.deque([], self.max_length)
            # buffer = []
        self.buffer = buffer

    @property
    def numberOfExperiences(self):
        # TODO optimize with caching/ check for modifying
        return len(self.buffer)

    @property
    def size(self):
        size = self.numberOfExperiences * Experience.size()
        return size

    def __len__(self):
        return len(self.buffer)

    def dequeue(self):
        # self.buffer.popleft()
        pass

    def prep(self, first_state):
        pass

    def is_full(self):
        return self.size > 42 or self.numberOfExperiences > self.max_length

    def is_ready(self):
        return self.numberOfExperiences >= self.start_length

    @property
    def states(self):
        #for item in self.buffer:
            #yield item.state
        return [item.state for item in self.buffer]
        # return [item.state for item in self.buffer]

    @property
    def actions(self):
        for item in self.buffer:
            yield item.action
        # return [item.action for item in self.buffer]

    @property
    def next_states(self):
        #for item in self.buffer:
            #yield item.next_state
         return [item.next_state for item in self.buffer]

    @property
    def rewards(self):
        for item in self.buffer:
            yield item.reward

    @property
    def is_dones(self):
        for item in self.buffer:
            yield item.isDone

    @property
    def training_items(self):
        for item in self.buffer:
            yield item.action, item.reward, item.isDone

    def append(self, experience):
        if self.is_full():
            self.dequeue()
        self.buffer.append(experience)

    def sample(self, sample_size):
        # return self.reservoirSampling(numberOfSamples)
        return self.randomSample(sample_size)

    def randomSample(self, numberOfSamples):
        # numpy choice is way slower than random.sample
        # sample_idxs = np.random.choice(range(len(self.buffer)), size=numberOfSamples)
        sample_idxs = random.sample(range(len(self.buffer)), numberOfSamples)
        samples = [self.buffer[idx] for idx in sample_idxs]
        return sample_idxs, ReplayBuffer(numberOfSamples, buffer=samples)

    def update(self, indexes, loss):
        pass

    def log(self):
        pass
        # 3logging.info(f"max buffer size: {self.max_length}")

    # Reservoir Sampling
    # TODO why did this have such POOR performance compared to random.sample????
    # Returns a sample of input data
    def reservoirSampling(self, numberOfSamples):
        sample = ReplayBuffer(numberOfSamples)
        for idx, experience in enumerate(self.buffer):
            if idx < numberOfSamples:
                sample.append(experience)
            elif idx >= numberOfSamples and random.random() < numberOfSamples / float(idx + 1):
                replace = random.randint(0, numberOfSamples - 1)
                sample[replace] = experience
        return sample


class AtariBuffer(ReplayBuffer):
    _all_states = None
    offset = 0
    state_idx = -1

    def __init__(self, max_length: int = 100000,
                 start_length: int = None,
                 buffer: list = None):
        ReplayBuffer.__init__(self, max_length, start_length, buffer)
        if AtariBuffer._all_states is None:
            AtariBuffer._all_states = collections.deque([], int(max_length * 1.1))

    def append(self, experience):
        # TODO force to be atari expereince to reduce slicing
        if self.is_full():
            self.dequeue()
        self.push_frame(experience.next_state)
        new_exp = Experience(None, experience.action, self.state_idx, experience.reward, experience.isDone)
        self.buffer.append(new_exp)
        del experience

    @property
    def size(self):
        replay_size = self.numberOfExperiences * Experience.size()
        atari_size = len(AtariBuffer._all_states) * AtariBuffer._all_states[0].size * AtariBuffer._all_states[
            0].itemsize / 1024 / 1024 / 1024
        return replay_size + atari_size

    @property
    def states(self):
        return (self.get_frames_from_idx(item.next_state-1) for item in self.buffer)

    @property
    def next_states(self):
        return (self.get_frames_from_idx(item.next_state) for item in self.buffer)

    def get_frames_from_idx(self, frame_idx):
        try:
            frames = np.stack([self._all_states[idx - AtariBuffer.offset] for idx in range(frame_idx-3, frame_idx+1)], axis=2)
        except IndexError:
            print("why....")
        return frames

    def randomSample(self, numberOfSamples):
        # numpy choice is way slower than random.sample
        # sample_idxs = np.random.choice(range(len(self.buffer)), size=numberOfSamples)
        sample_idxs = random.sample(range(len(self.buffer)), numberOfSamples)
        samples = [self.buffer[idx] for idx in sample_idxs]
        # TODO stop override
        return sample_idxs, AtariBuffer(numberOfSamples, buffer=samples)

    def push_frame(self, state: np.array):
        assert(len(state.shape) < 3, "incorrect shape")
        pre_len = len(self._all_states)
        self._all_states.append(state)
        self.state_idx += 1
        post_len = len(self._all_states)
        if pre_len == post_len:
            AtariBuffer.offset += 1

    def prep(self, state):
        self.push_frame(state)
        self.push_frame(state)
        self.push_frame(state)
        self.push_frame(state)


class SampleBuffer(ReplayBuffer):
    @property
    def states(self):
        return [self.get_frames_from_idx(item.next_state-1) for item in self.buffer]

    @property
    def next_states(self):
        return [self.get_frames_from_idx(item.next_state) for item in self.buffer]

    def get_frames_from_idx(self, frame_idx):
        try:
            frames = np.stack([self._all_states[idx - AtariBuffer.offset] for idx in range(frame_idx-3, frame_idx+1)], axis=2)
        except IndexError:
            print("why....")
        return frames

