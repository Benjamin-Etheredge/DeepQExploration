import collections
from collections import deque
import random
import sys
from PIL import Image

import numpy as np

from numpy import dstack, mean, stack, array


# TODO implement priority replay buffer

class VoidBuffer:
    def __init__(self):
        pass

class Experience:
    MEMORY_SIZE = None

    #@profile
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
        self._reward = reward
        self._isDone = is_done
        '''
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
                                      sys.getsizeof(self._action) + sys.getsizeof(self._reward) +
                                      sys.getsizeof(self.__isDone)) / 1024. / 1024. / 1024.
        '''

    '''
    @classmethod
    @profile
    def size(cls):
        return cls.MEMORY_SIZE
    '''

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
    def isDone(self):
        return self._isDone


class ExperienceLists(Experience):

    @property
    def state(self):
        return np.stack(self._state, axis=2)

    @property
    def next_state(self):
        return np.stack(self._next_state, axis=2)




class AtariExperience(Experience):

    def __init__(self, state, action, next_state, reward, is_done):
        Experience.__init__(self, state, action, next_state[:, :, -1], reward, is_done)

    @property
    def next_state(self):
        return dstack((self._state[:, :, 1:], self._next_state))

    @staticmethod
    def gray_scale(img):
        return mean(array(img), axis=2)[::2, ::2].astype(np.uint8)  # TODO reduce 3 -> 2


def clipped_attari(reward, *args, **kwargs):
    reward = min(1, max(reward, -1))
    return AtariExperience(reward=reward, *args, **kwargs)


class ReplayBuffer:

    def __init__(self,
                 max_length: int = 100000,
                 start_length: int = None,
                 buffer = None):
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

        # TODO deque may be slow for sampling

        # buffer = []
        if buffer is not None:
            self.buffer = deque([], len(buffer))
            self.buffer.extend(buffer)
        else:
            self.buffer = deque([], self.max_length)

    @property
    def numberOfExperiences(self):
        # TODO optimize with caching/ check for modifying
        return len(self.buffer)

    '''
    @property
    def size(self):
        size = self.numberOfExperiences * Experience.size()
        return size
    '''

    def __len__(self):
        return len(self.buffer)

    def dequeue(self):
        #self.buffer.popleft()
        #count1 = sys.getrefcount(self.buffer[0])
        #item = self.buffer.popleft()
        #count2 = sys.getrefcount(item)
        #del item
        pass

    def prep(self, first_state):
        pass

    def is_full(self):
        return self.numberOfExperiences >= self.max_length

    def is_ready(self):
        return self.numberOfExperiences >= self.start_length

    @property
    def states(self):
        #for item in self.buffer:
            #yield item.state
        #return np.stack([item.state for item in self.buffer], axis=2)
        return [np.stack(item.state, axis=2) for item in self.buffer]
        # return [item.state for item in self.buffer]

    @property
    def actions(self):
        for item in self.buffer:
            yield item.action
        # return [item.action for item in self.buffer]

    @property
    def next_states(self):
        # No reason to use a generator here as keras would require it to be converted to a list beforehand
        return [np.stack(item.next_state, axis=2) for item in self.buffer]

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
        del experience

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
            AtariBuffer._all_states = collections.deque([], int(max_length * 1.01))

    #@profile
    def append(self, experience):
        # TODO force to be atari expereince to reduce slicing
        if self.is_full():
            self.dequeue()
        self.push_frame(experience.next_state[:, :, -1])
        #new_exp = Experience(None, experience.action, self.state_idx, experience.reward, experience.isDone)
        new_exp = (experience.action, self.state_idx, experience.reward, experience.isDone)
        self.buffer.append(new_exp)
        del experience

    '''
    @property
    #@profile
    def size(self):
        replay_size = self.numberOfExperiences * Experience.size()
        atari_size = len(AtariBuffer._all_states) * AtariBuffer._all_states[0].size * AtariBuffer._all_states[
            0].itemsize / 1024 / 1024 / 1024
        return replay_size + atari_size
    '''

    @property
    #@profile
    def states(self):
        return [self.get_frames_from_idx(item[1]-1) for item in self.buffer]
        #return (self.get_frames_from_idx(item.next_state-1) for item in self.buffer)

    @property
    def actions(self):
        return [item[0] for item in self.buffer]

    @property
    #@profile
    def next_states(self):
        #return (self.get_frames_from_idx(item.next_state) for item in self.buffer)
        return (self.get_frames_from_idx(item[1]) for item in self.buffer)

    @property
    def rewards(self):
        return [item[2] for item in self.buffer]

    @property
    def is_dones(self):
        return [item[3] for item in self.buffer]
    # 8.569GiB before tuple buffer

    #@profile
    def get_frames_from_idx(self, frame_idx):
        try:
            frames = stack([self._all_states[idx - AtariBuffer.offset] for idx in range(frame_idx-3, frame_idx+1)], axis=2)
        except IndexError:
            print("why....")
        return frames

    #@profile
    def randomSample(self, numberOfSamples):
        # numpy choice is way slower than random.sample
        # sample_idxs = np.random.choice(range(len(self.buffer)), size=numberOfSamples)
        sample_idxs = random.sample(range(len(self.buffer)), numberOfSamples)
        samples = [self.construct_experience(idx) for idx in sample_idxs]
        # TODO stop override
        return sample_idxs, ReplayBuffer(numberOfSamples, buffer=samples)

    #@profile
    def construct_experience(self, idx: int):
        #experienc: Experience = self.buffer[idx]
        experienc = self.buffer[idx]
        frame_idx: int = experienc[1]
        return Experience(self.get_frames_from_idx(frame_idx-1),
                          experienc[0],
                          self.get_frames_from_idx(frame_idx),
                          experienc[2],
                          experienc[3])

    @classmethod
    #@profile
    def push_frame(cls, state: array):
        assert len(state.shape) < 3
        pre_len = len(cls._all_states)
        cls._all_states.append(state)
        cls.state_idx += 1
        post_len = len(cls._all_states)
        if pre_len == post_len:
            AtariBuffer.offset += 1

    #@profile
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
    #@profile
    def next_states(self):
        return [self.get_frames_from_idx(item.next_state) for item in self.buffer]

    #@profile
    def get_frames_from_idx(self, frame_idx):
        try:
            frames = stack([self._all_states[idx - AtariBuffer.offset] for idx in range(frame_idx-3, frame_idx+1)], axis=2)
        except IndexError:
            print("why....")
        return frames

