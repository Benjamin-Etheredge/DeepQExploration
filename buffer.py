import numpy as np
import logging
import random
import collections


#TODO implement priority replay buffer

class Experience:
    def __init__(self, state, action, nextState, reward, isDone):
        self.data = [state, action, nextState, reward, isDone]
        self.__state = np.array(state)
        self.__action = action
        self.__nextState = np.array(nextState)
        self.__reward = reward
        self.__isDone = isDone

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
        self.buffer = buffer

    @property
    def numberOfExperiences(self):
        logging.debug('numberOfExperiences')
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
        logging.debug('dequeue')
        #self.buffer = self.buffer[1:]
        pass

    def isFull(self):
        logging.debug('isReplayBufferFull')
        return self.numberOfExperiences > self.max_length

    def isReady(self):
        logging.debug('isReplayBuffer')
        return self.numberOfExperiences >= self.start_length

    @property
    def states(self):
        return np.array([item.state for item in self.buffer])

    @property
    def actions(self):
        return [item.action for item in self.buffer]

    @property
    def nextStates(self):
        return np.array([item.nextState for item in self.buffer])

    @property
    def rewards(self):
        return [item.reward for item in self.buffer]

    @property
    def isDones(self):
        return [item.isDone for item in self.buffer]

    def append(self, experience):
        if self.isFull():
            self.dequeue()
        self.buffer.append(experience)

    def sample(self, sample_size=None):
        # return self.reservoirSampling(numberOfSamples)
        if sample_size is None:
            sample_size = self.sample_size
        return self.randomSample(sample_size)

    def randomSample(self, numberOfSamples):
        return ReplayBuffer(numberOfSamples, buffer=random.sample(self.buffer, numberOfSamples))

    def log(self):
        print("info - start buffer size: {0}".format(self.start_length))
        print("info - max buffer size: {0}".format(self.max_length))

    # Reservoir Sampling
    #TODO why did this have such POOR performance compared to random.sample????
    # Returns a sample of input data
    def reservoirSampling(self, numberOfSamples):
        logging.debug('Sampling')
        sample = ReplayBuffer(numberOfSamples)
        for idx, experience in enumerate(self.buffer):
            if idx < numberOfSamples:
                sample.append(experience)
            elif idx >= numberOfSamples and random.random() < numberOfSamples / float(idx + 1):
                replace = random.randint(0, numberOfSamples - 1)
                sample[replace] = experience
        return sample

