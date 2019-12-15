import numpy as np
import random
import collections


#TODO implement priority replay buffer

class Experience:
    def __init__(self, state, action, next_state, reward, is_done):
        self.data = [state, action, next_state, reward, is_done]
        self.__state = np.array(state)
        self.__action = action
        self.__nextState = np.array(next_state)
        self.__reward = reward
        self.__isDone = is_done

    def __repr__(self):
        return self.data

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value
        return value

    @property
    def state(self):
        return self.__state

    @property
    def action(self):
        return self.__action

    @property
    def nextState(self):
        return self.__nextState

    @property
    def reward(self):
        return self.__reward

    @property
    def isDone(self):
        return self.__isDone


class ReplayBuffer:

    def __init__(self,
                 max_length: int = 100000,
                 start_length: int =None,
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
        return self.numberOfExperiences > self.max_length

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
        return np.array([item.nextState for item in self.buffer])

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
