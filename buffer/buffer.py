import numpy as np

np.random.seed(4)

import random
random.seed(4)


# TODO implement priority replay buffer

class ReplayBuffer:

    def __init__(self,
                 max_length: int = 100000,
                 start_length: int = None,
                 buffer=None):

        self.max_length = max_length

        if start_length is None:
            start_length = max_length
        self.start_length = start_length

        # Switched form deque due to slow indexing
        if buffer is not None:
            #self.buffer = deque([], len(buffer))
            #self.buffer.extend(buffer)
            self.buffer = list(buffer)
        else:
            #self.buffer = deque([], self.max_length)
            self.buffer = []

    @property
    def experience_count(self):
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def dequeue(self):
        self.buffer.pop(0)

    def prep(self, first_state):
        pass

    def is_full(self):
        return self.experience_count >= self.max_length

    def is_ready(self):
        return self.experience_count >= self.start_length

    def training_items(self):
        states = []
        actions = []
        next_states = []
        rewards = []
        is_dones = []
        #items = [[np.stack(item.state, axis=2), item.action, np.stack(item.next_state, axis=2), item.reward, item.isDone] for item in self.buffer]
        #items = [array(value) for value in items]
        #return *items
        for item in self.buffer:
            states.append(item.state)
            actions.append(item.action)
            next_states.append(item.next_state)
            rewards.append(item.reward)
            is_dones.append(item.isDone)
        return np.array(states, dtype=np.uint8), np.array(actions, dtype=np.uint8), np.array(next_states, dtype=np.uint8), \
               np.array(rewards), np.array(is_dones, dtype=np.bool_)

    def append(self, experience):
        if self.is_full():
            self.dequeue()
        self.buffer.append(experience)

    def sample(self, sample_size):
        # return self.reservoirSampling(numberOfSamples)
        return self.random_sample(sample_size)

    def random_sample(self, sample_count):
        # numpy choice is way slower than random.sample
        # sample_idxs = np.random.choice(range(len(self.buffer)), size=numberOfSamples)
        sample_idxs = random.sample(range(len(self.buffer)), sample_count)
        samples = [self.buffer[idx] for idx in sample_idxs]
        return sample_idxs, ReplayBuffer(sample_count, buffer=samples)
        #samples = np.random.choice(self.buffer, size=sample_count, replace=False)
        #return None, ReplayBuffer(sample_count, buffer=samples)

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
            elif idx >= numberOfSamples and np.random.random() < numberOfSamples / float(idx + 1):
                replace = np.random.randint(0, numberOfSamples - 1)
                sample[replace] = experience
        return sample

    # https://medium.com/ibm-watson/incredibly-fast-random-sampling-in-python-baf154bd836a
    def multidimensional_shifting(num_samples, sample_size, elements, probabilities=None):
        if probabilities is None:
            probabilities = np.tile(1/num_samples, (num_samples, 1))
        # replicate probabilities as many times as `num_samples`
        replicated_probabilities = np.tile(probabilities, (num_samples, 1))
        # get random shifting numbers & scale them correctly
        random_shifts = np.random.random(replicated_probabilities.shape)
        random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
        # shift by numbers & find largest (by finding the smallest of the negative)
        shifted_probabilities = random_shifts - replicated_probabilities
        return np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]