import numpy as np
import logging
import random


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

    def __init__(self, maxLength, startLength=None, buffer=None):
        if buffer is None:
            self.buffer = []
        else:
            self.buffer = buffer
        self.maxLength = maxLength
        if startLength is None:
            self.startLength = maxLength
        else:
            self.startLength = startLength

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

    def dequeueFromReplayBuffer(self):
        logging.debug('dequeueFromReplayBuffer')
        self.buffer = self.buffer[1:]

    def isFull(self):
        logging.debug('isReplayBufferFull')
        return self.numberOfExperiences > self.maxLength

    def isReady(self):
        logging.debug('isReplayBuffer')
        return self.numberOfExperiences > self.startLength

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
            self.dequeueFromReplayBuffer()
        self.buffer.append(experience)

    def sample(self, numberOfSamples):
        # return self.reservoirSampling(numberOfSamples)
        return self.randomSample(numberOfSamples)

    def randomSample(self, numberOfSamples):
        return ReplayBuffer(numberOfSamples, buffer=random.sample(self.buffer, numberOfSamples))

    def log(self):
        print("info - start buffer size: {0}".format(self.startLength))
        print("info - max buffer size: {0}".format(self.maxLength))

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

