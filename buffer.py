from collections import deque
import random

import numpy as np

from numpy import mean, array


# TODO implement priority replay buffer
from experience import Experience


class VoidBuffer:
    def __init__(self):
        pass

class AtariExperience(Experience):

    @staticmethod
    def gray_scale(img):
        return mean(array(img), axis=2)[::2, ::2].astype(np.uint8)  # TODO reduce 3 -> 2


def clipped_atari(reward, *args, **kwargs):
    reward = min(1, max(reward, -1))
    return AtariExperience(reward=reward, *args, **kwargs)


class ReplayBuffer:

    def __init__(self,
                 max_length: int = 100000,
                 start_length: int = None,
                 buffer=None):

        self.max_length = max_length

        if start_length is None:
            start_length = max_length
        self.start_length = start_length

        # TODO deque may be slow for sampling

        if buffer is not None:
            self.buffer = deque([], len(buffer))
            self.buffer.extend(buffer)
        else:
            self.buffer = deque([], self.max_length)

    @property
    def experience_count(self):
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def dequeue(self):
        pass

    def prep(self, first_state):
        pass

    def is_full(self):
        return self.experience_count >= self.max_length

    def is_ready(self):
        return self.experience_count >= self.start_length

    @property
    def states(self):
        return [np.stack(item.state, axis=2) for item in self.buffer]
        # return [item.state for item in self.buffer]

    @property
    def actions(self):
        for item in self.buffer:
            yield item.action

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
        return self.random_sample(sample_size)

    def random_sample(self, sample_count):
        # numpy choice is way slower than random.sample
        # sample_idxs = np.random.choice(range(len(self.buffer)), size=numberOfSamples)
        sample_idxs = random.sample(range(len(self.buffer)), sample_count)
        samples = [self.buffer[idx] for idx in sample_idxs]
        return sample_idxs, ReplayBuffer(sample_count, buffer=samples)

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
