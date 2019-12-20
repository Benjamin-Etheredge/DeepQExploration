import numpy as np
import random
import collections
import sys


#TODO implement priority replay buffer

class Experience:
    MEMORY_SIZE = 0
    def __init__(self, state, action, next_state, reward, is_done):
        #self.data = [state, action, next_state, reward, is_done]
        self._state = np.squeeze(np.array(state))
        #if len(self.__state.shape) > 1:
            #self.__state = self.__state.flatten()
        self._action = action
        self._next_state = np.squeeze(np.array(next_state))
        #if len(self.__nextState.shape) > 1:
            #self.__nextState = self.__nextState.flatten()
        self.__reward = reward
        self.__isDone = is_done
        if Experience.MEMORY_SIZE == 0:
            Experience.MEMORY_SIZE = ((self._state.size * self._state.itemsize) +
                                      (self._next_state.size * self._next_state.itemsize) +
                                      sys.getsizeof(self._action) + sys.getsizeof(self.__reward) +
                                      sys.getsizeof(self.__isDone)) / 1024. / 1024. / 1024.

    #def __repr__(self):
        #return self.data

    #def __getitem__(self, item):
        #return self.data[item]

    #def __setitem__(self, key, value):
        #self.data[key] = value
        #return value

    @staticmethod
    def size():
        return Experience.MEMORY_SIZE

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
        Experience.__init__(self, np.array(state, dtype=np.uint8), action, np.array(next_state[-1], dtype=np.uint8), reward, is_done)

    @property
    def next_state(self):
        #temp =  self._state[1:, :, :]
        #temp2 =  self._state[1:]
        #next =  self._state[1:] + self._next_state
        #next3 =  np.vstack([self._state[1:], self._next_state[np.newaxis, :, :]])
        #next2 =  np.hstack([self._state[1:], self._next_state[np.newaxis, :, :]])
        #return next3
        return np.vstack([self._state[1:], self._next_state[np.newaxis, :, :]])

    #@staticmethod
    #def gray_scale(x):
        #x = x[::2, ::2]
        #gray = np.array((0.21 * x[:, :, :1]) + (0.72 * x[:, :, 1:2]) + (0.07 * x[:, :, -1:]), dtype=np.uint8)
        #return gray

    @staticmethod
    def gray_scale(img):
        img = img[::2, ::2]
        return np.mean(img, axis=2).astype(np.uint8) # TODO reduce 3 -> 2
        #return temp[:, :, np.newaxis]
        #return img[:, :, 1]


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
            #buffer = []
        self.buffer = buffer

        self._state_cache = None
        self._next_state_cache = None
        self._action_cache = None
        self._is_done_cache = None

    @property
    def numberOfExperiences(self):
        # TODO optimize with caching/ check for modifying
        return len(self.buffer)

    def size(self):
        size = self.numberOfExperiences * Experience.MEMORY_SIZE
        return size

    def __setitem__(self, key, value):
        self.buffer[key] = value
        return self.buffer[key]

    # def __getitem__(self, item):
    # return self.buffer[item]

    def __len__(self):
        return len(self.buffer)

    def dequeue(self):
        self.buffer.popleft()
        pass

    def is_full(self):
        return self.size() > 42 or self.numberOfExperiences > self.max_length

    def is_ready(self):
        return self.numberOfExperiences >= self.start_length

    @property
    def states(self):
        return np.array([item.state for item in self.buffer])

    @property
    def actions(self):
        return [item.action for item in self.buffer]
        #for item in self.buffer:
            #yield item.action

    @property
    def next_states(self):
        return np.array([item.next_state for item in self.buffer])

    @property
    def rewards(self):
        for item in self.buffer:
            yield item.reward

    @property
    def is_dones(self):
        #return [item.isDone for item in self.buffer]
        for item in self.buffer:
            yield item.isDone

    @property
    def training_items(self):
        # self.data[1] = [state, action, next_state, reward, is_done]
        #return self.data[1], self.data[3], self.data[4]
        for item in self.buffer:
            yield (item.action, item.reward, item.isDone)

    def append(self, experience):
        if self.is_full():
            self.dequeue()
        self.buffer.append(experience)

    def sample(self, sample_size):
        # return self.reservoirSampling(numberOfSamples)
        return self.randomSample(sample_size)

    def randomSample(self, numberOfSamples):
        # numpy choice is way slower than random.sample
        #sample_idxs = np.random.choice(range(len(self.buffer)), size=numberOfSamples)
        sample_idxs = random.sample(range(len(self.buffer)), numberOfSamples)
        samples = [self.buffer[idx] for idx in sample_idxs]
        return sample_idxs, ReplayBuffer(numberOfSamples, buffer=samples)

    def update(self, indexs, loss):
        pass

    def log(self):
        pass
        #3logging.info(f"max buffer size: {self.max_length}")

    # Reservoir Sampling
    #TODO why did this have such POOR performance compared to random.sample????
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
