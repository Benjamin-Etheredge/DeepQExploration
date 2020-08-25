import numpy as np

np.random.seed(4)

import random
random.seed(4)
from collections import deque
#from collections import OrderedDict # TODO remove, as of 3.6, dicts are ordered

OFFSET = 0.01
DEFAULT_PRIO = 999999999
BETA_INCREMENT  = 0.000025
ALPHA_INCREMENT = 0.000025



# TODO implement priority replay buffer

class ReplayBuffer:

    def __init__(self,
                 max_length: int,
                 start_length: int,
                 buffer=None,
                 priorities=None,
                 probabilities=None):

        assert max_length >= start_length, "Max length must be greater than or equal to start length"
        self.max_length: int = max_length

        if start_length is None:
            start_length: int = max_length
        self.start_length: int = start_length
        self.total_items_added = 0
        self.oldest_item_idx = 0

        # Switched form deque due to slow indexing
        #indices = [idx for idx in range(max_length)]
        if buffer is not None:
            #self.buffer = deque([], len(buffer))
            #self.buffer.extend(buffer)
            # TODO time ordered dict vs regular
            #self.buffer = OrderedDict({idx: item for idx, item in enumerate(buffer)})
            self.buffer = {idx: item for idx, item in enumerate(buffer)}
            #default_prio = 1
            #self.priorities= OrderedDict({idx: prio for idx, prio in enumerate(priorities)})
            self.priorities = {idx: prio for idx, prio in enumerate(priorities)}
            self.probabilities = probabilities
            #self.buffer: list = list(buffer)
        else:
            #self.buffer = deque([], self.max_length)
            #self.buffer = OrderedDict()
            #self.priorities= OrderedDict()
            self.buffer = {}
            self.priorities= {}
            self.priorities = np.zeros(max_length)
            self.probabilities = np.zeros(max_length)
            #self.buffer: list = []
        #self.total_priority = sum(self.priorities.values())
        #self.total_priority = sum(self.priorities)
        self.alpha = 0.6
        self.beta = 0.4

    @property
    def experience_count(self):
        return self.total_items_added - self.oldest_item_idx

    @property
    def states(self):
        states = [item.state for item in self.buffer]
        frame_count = len(states[1])
        states = [np.array([state[frame_idx] for state in states], dtype=np.uint8) for frame_idx in range(frame_count)]
        return states
        
    def __len__(self):
        return self.experience_count

    def dequeue(self):
        if self.experience_count > 0:  # if the list is empty, that's fine
            del self.buffer[self.oldest_item_idx]
            self.oldest_item_idx += 1

    def prep(self, first_state):
        pass

    def is_full(self):
        return self.experience_count >= self.max_length

    def is_ready(self):
        return self.experience_count >= self.start_length

    # TODO move this logic to agent
    def training_items(self):
        #states = []
        actions = []
        #next_states = []
        rewards = []
        is_dones = []
        # No difference in execution speed
        states = [[] for _ in range(4)]
        next_states = [[] for _ in range(4)]
        for item in self.buffer.values():
            frame_count = len(item.state)
            for frame_idx in range(frame_count):
                frame = item.state[frame_idx]
                states[frame_idx].append(frame)
            #states.append(all_state)
            actions.append(item.action)


            for frame_idx in range(frame_count):
                frame = item.next_state[frame_idx]
                next_states[frame_idx].append(frame)
            #next_states.append(item.next_state)
            rewards.append(item.reward)
            is_dones.append(item.is_done)

        #states = [np.array([state[frame_idx] for state in states], dtype=np.uint8) for frame_idx in range(frame_count)]
        actions = np.array(actions, dtype=np.uint8)
        #next_states = [np.array([state[frame_idx] for state in next_states], dtype=np.uint8) for frame_idx in range(frame_count)]
        rewards = np.array(rewards, dtype=np.float32)
        is_dones = np.array(is_dones, dtype=np.bool_)

        return states, actions, next_states, rewards, is_dones

    def append(self, experience):
        if self.is_full():
            self.dequeue()
        #self.buffer.append(experience)
        self.buffer[self.total_items_added] = experience
        self.priorities[self.total_items_added%self.max_length] = DEFAULT_PRIO 
        self.probabilities[self.total_items_added%self.max_length] = DEFAULT_PRIO ** self.alpha
        self.total_items_added += 1

    def sample(self, sample_size):
        # return self.reservoirSampling(numberOfSamples)
        return self.random_sample(sample_size)
        #return multidimensional_shifting(self.example_count, sample_size, range(self.experience_count), probabilities=self.probabilities):

    # TODO consider changing from dict to numpy array
    @property
    def prob(self):
        probs = self.probabilities[:self.experience_count]
        #total = np.sum(probs)
        #probs = np.power(probs, self.alpha)  # testing storing this
        probs /= np.sum(probs)
        #return [(prob**self.alpha)/(damper) for prob in self.priorities.values()] # this is super slow
        return probs

    def random_indices(self, sample_count):
        #return random.sample(range(self.experience_count), sample_count)
        #return np.random.randint(low=0, high=self.experience_count, size=sample_count) # same speed as above
        #return random.choices(range(self.experience_count), weights=self.prob, k=sample_count)
        prob = self.prob
        #return np.random.choice(range(self.experience_count), replace=False, p=self.prob, size=sample_count), prob
        return np.squeeze(multidimensional_shifting(1, sample_count, range(self.experience_count), probabilities=prob)), prob

    def random_sample(self, sample_count):
        # numpy choice is way slower than random.sample
        # sample_idxs = np.random.choice(range(len(self.buffer)), size=numberOfSamplesk)
        #sample_idxs = random.sample(range(self.experience_count), sample_count)
        sample_idxs, probs = self.random_indices(sample_count)
        #print(sample_idxs)
        #sample_idxs = list(range(sample_count))
        samples = [self.buffer[idx+self.oldest_item_idx] for idx in sample_idxs]
        prios = [self.priorities[(idx+self.total_items_added) % self.max_length] for idx in sample_idxs]
        #damper = 1/self.experience_count
        # TODO add beta term
        weights = (self.experience_count * probs) ** -self.beta
        # TODO break out a simpier experience holder interface
        return sample_idxs, weights, ReplayBuffer(sample_count, sample_count, buffer=samples, priorities=prios)
        #samples = np.random.choice(self.buffer, size=sample_count, replace=False)
        #return None, ReplayBuffer(sample_count, buffer=samples)

    def update(self, indexes, loss):
        # TODO make losses sequence of losses
        for idx in indexes:
            #self.total_priority -= self.priorities[idx+self.oldest_item_idx]
            #self.priorities[idx] = ((loss**self.alpha) / (self.total_priority**self.alpha))
            #self.priorities[idx] = ((loss**self.alpha) / ([prio**self.alpha for prio in self.priorities.values()])
            #self.total_priority += self.priorities[idx+self.oldest_item_idx]
            #self.priorities[(idx+self.total_items_added)%self.max_length] = loss
            self.priorities[(idx+self.oldest_item_idx)%self.max_length] = loss
            #self.probabilities[(idx+self.total_items_added)%self.max_length] = (loss+OFFSET) ** self.alpha
            self.probabilities[(idx+self.oldest_item_idx)%self.max_length] = (loss+OFFSET) ** self.alpha
        self.alpha = min(1, self.alpha+ALPHA_INCREMENT)
        self.beta = min(1, self.beta+BETA_INCREMENT)

    '''
    @property
    def weights(self):
        try:
            damper = 1/self.experience_count
        except ZeroDivisionError:
            damper = 1
        # TODO this is not sure to preserve order of idxs, rework
        return [ damper*(1/prio) for idx, prio in self.priorities.items()]
    '''

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
    

# https://adventuresinmachinelearning.com/sumtree-introduction-python/
class Node:
    def __init__(self, left, right, is_leaf: bool = False, idx = None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        if not self.is_leaf:
            self.value = self.left.value + self.right.value
        self.parent = None
        self.idx = idx  # this value is only set for leaf nodes
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self

    @classmethod
    def create_leaf(cls, value, idx):
        leaf = cls(None, None, is_leaf=True, idx=idx)
        leaf.value = value
        return leaf

def create_tree(input: list):
    nodes = [Node.create_leaf(v, i) for i, v in enumerate(input)]
    leaf_nodes = nodes
    while len(nodes) > 1:
        inodes = iter(nodes)
        nodes = [Node(*pair) for pair in zip(
            inodes, inodes)]
    return nodes[0], leaf_nodes

def retrieve(value: float, node: Node):
    if node.is_leaf:
        return node
    if node.left.value >= value:
        return retrieve(value, node.left)
    else:
        return retrieve(value - node.left.value, node.right)

def update(node: Node, new_value: float):
    change = new_value - node.value
    node.value = new_value
    propagate_changes(change, node.parent)

def propagate_changes(change: float, node: Node):
    node.value += change
    if node.parent is not None:
        propagate_changes(change, node.parent)