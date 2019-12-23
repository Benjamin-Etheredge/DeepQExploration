import numpy as np
import random
import collections
import sys


#TODO implement priority replay buffer

class Experience:
    MEMORY_SIZE = 0
    def __init__(self, state, action, next_state, reward, is_done):
        #self.data = [state, action, next_state, reward, is_done]
        self._state = state
        #print(self._state.shape)
        #if len(self.__state.shape) > 1:
            #self.__state = self.__state.flatten()
        self._action = action
        self._next_state = next_state
        #if len(self.__nextState.shape) > 1:
            #self.__nextState = self.__nextState.flatten()
        self.__reward = reward
        self.__isDone = is_done
        if Experience.MEMORY_SIZE == 0:
            Experience.MEMORY_SIZE = ((self._state.size * self._state.itemsize) +
                                      #(self._next_state.size * self._next_state.itemsize) +
                                      sys.getsizeof(self._action) + sys.getsizeof(self.__reward) +
                                      sys.getsizeof(self.__isDone)) / 1024. / 1024. / 1024.

    #def __repr__(self):
        #return self.data

    #def __getitem__(self, item):
        #return self.data[item]

    #def __setitem__(self, key, value):
        #self.data[key] = value
        #return value

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
        Experience.__init__(self,
                            np.array(np.concatenate((state, next_state[:, :, -1:]), axis=2), dtype=np.uint8),
                            action, None, reward, is_done)

    @property
    def state(self):
        temp = self._state[:, :, :-1]
        return temp

    @property
    def next_state(self):
        temp = self._state[:, :, 1:]
        return temp

    @staticmethod
    def gray_scale(img):
        img = img[::2, ::2]
        return np.mean(img, axis=2).astype(np.uint8)  # TODO reduce 3 -> 2

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

    def __len__(self):
        return len(self.buffer)

    def dequeue(self):
        #self.buffer.popleft()
        pass

    def prep(self, first_state):
        pass

    def is_full(self):
        return self.size() > 42 or self.numberOfExperiences > self.max_length

    def is_ready(self):
        return self.numberOfExperiences >= self.start_length

    @property
    def states(self):
        for item in self.buffer:
            yield item.state
        #return [item.state for item in self.buffer]
        #return [item.state for item in self.buffer]

    @property
    def actions(self):
        for item in self.buffer:
           yield item.action
        #return [item.action for item in self.buffer]

    @property
    def next_states(self):
        for item in self.buffer:
            yield item.next_state
        #return [item.next_state for item in self.buffer]

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
        #sample_idxs = np.random.choice(range(len(self.buffer)), size=numberOfSamples)
        sample_idxs = random.sample(range(len(self.buffer)), numberOfSamples)
        samples = [self.buffer[idx] for idx in sample_idxs]
        return sample_idxs, ReplayBuffer(numberOfSamples, buffer=samples)

    def update(self, indexes, loss):
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

class AtariBuffer(ReplayBuffer):
    _all_states = None
    offset = 0
    state_idx = 0
    def __init__(self, max_length: int = 100000,
                    start_length: int = None,
                    buffer: list = None):
        ReplayBuffer.__init__(self, max_length, start_length, buffer)
        if AtariBuffer._all_states is None:
            AtariBuffer._all_states = collections.deque([], int(max_length*1.1))

    def clear_cache(self):
        self._next_state_cache = None
        self._state_cache = None


    def append(self, experience):
        self.clear_cache()
        temp = len(AtariBuffer._all_states)
        AtariBuffer._all_states.append(experience.next_state[:, :, -1])
        temp2 = len(AtariBuffer._all_states)

        if temp == temp2:
            self.offset += 1
        new_exp = Experience(None, experience.action, self.state_idx, experience.reward, experience.isDone)
        self.buffer.append(new_exp)
        self.state_idx += 1

    @classmethod
    def size(cls):
        return len(cls._all_states) * Experience.size()

    @property
    def states(self):
        #temp = [self._all_states[item.next_state-4: item.next_state-1] for item in self.buffer]
        #temp = [np.stack([AtariBuffer._all_states[idx-self.offset] for idx in range(item.next_state-4, item.next_state-1)], axis=2) for item in self.buffer]
        #temp = []
        #for item in self.buffer:
            #temp2 = []
            #for idx in range(item.next_state-4, item.next_state-1):
                #value = AtariBuffer._all_states[idx-self.offset]
                #temp2.append(value)
        return (np.stack([AtariBuffer._all_states[idx-self.offset] for idx in range(item.next_state-4, item.next_state)], axis=2) for item in self.buffer)
            #temp.append(temp2)
        #Jkjif self._state_cache is None:
            #Jkjself._state_cache = (np.stack([AtariBuffer._all_states[idx-self.offset] for idx in range(item.next_state-4, item.next_state)], axis=2) for item in self.buffer)
        #Jkjreturn self._state_cache

    @property
    def next_states(self):
        #return [self._allstates[state_idxs] for state_idxs in self._next_states]
        #return [item.next_state for item in self.buffer]
        return (np.stack([AtariBuffer._all_states[idx-self.offset] for idx in range((item.next_state-3), item.next_state+1)], axis=2) for item in self.buffer)
        #if self._next_state_cache is None:
            #self._next_state_cache = (np.stack([AtariBuffer._all_states[idx-self.offset] for idx in range((item.next_state-3), item.next_state+1)], axis=2) for item in self.buffer)
        #return self._next_state_cache

    @property
    def training_items(self):
        for item in self.buffer:
            yield (item.action, item.reward, item.isDone)

    def get_samples_from_idxs(idxs):
        pass

    def randomSample(self, numberOfSamples):
        # numpy choice is way slower than random.sample
        #sample_idxs = np.random.choice(range(len(self.buffer)), size=numberOfSamples)
        sample_idxs = random.sample(range(len(self.buffer)), numberOfSamples)
        samples = [self.buffer[idx] for idx in sample_idxs]
        #samples = [Experience(sample. for sample in samples]
        return sample_idxs, AtariBuffer(numberOfSamples, buffer=samples)

    def prep(self, state):
        AtariBuffer._all_states.append(state)
        self.state_idx += 1
        AtariBuffer._all_states.append(state)
        self.state_idx += 1
        AtariBuffer._all_states.append(state)
        self.state_idx += 1
        AtariBuffer._all_states.append(state)
        self.state_idx += 1

