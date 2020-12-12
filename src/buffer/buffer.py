import numpy as np

np.random.seed(4)

import random
random.seed(4)
from collections import deque
#from collections import OrderedDict # TODO remove, as of 3.6, dicts are ordered

OFFSET = 0.01
DEFAULT_PRIO = 1
BETA_INCREMENT  = 0.000025
ALPHA_INCREMENT = 0.000025

class ReplayBuffer:

    def __init__(self,
                 max_length: int,
                 start_length: int,
                 alpha=0.6,
                 beta=0.4,
                 alpha_inc=0.000025,
                 beta_inc=0.000025):

        assert max_length >= start_length, "Max length must be greater than or equal to start length"
        self.max_length: int = max_length

        if start_length is None:
            start_length: int = max_length
        self.start_length: int = start_length
        self.total_items_added = 0
        self.oldest_item_idx = 0

        self.buffer = {}
        self.root_node, self.leaf_nodes = create_tree(np.zeros(max_length))
        self.alpha = alpha
        self.beta = beta
        self.alpha_inc = alpha_inc
        self.beta_inc = beta_inc


    @property
    def experience_count(self):
        return self.total_items_added - self.oldest_item_idx

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

    def append(self, experience):
        if self.is_full():
            self.dequeue()

        self.buffer[self.total_items_added] = experience

        item_idx = self.total_items_added%self.max_length
        update(self.leaf_nodes[item_idx], DEFAULT_PRIO**self.alpha)

        self.total_items_added += 1
    
    def node_sample_indicies(self, sample_size):
        random_values = np.random.uniform(0, self.root_node.value, size=sample_size)
        probs = np.zeros(len(random_values))
        indicies = np.zeros(len(random_values), dtype=np.uint32) # dtype prevents floating point indicies
        for store_idx, value in enumerate(random_values):
            node = retrieve(value, self.root_node)
            probs[store_idx] = node.value 
            indicies[store_idx] = node.idx
        probs /= self.root_node.value
        weights = (self.experience_count * probs) ** -self.beta
        return indicies, weights

    def translate_idx(self, idx):
        zeroth_idx = self.total_items_added%self.max_length
        return idx+zeroth_idx

    def random_sample_node(self, sample_size):
        indicies, weights = self.node_sample_indicies(sample_size)
        samples = [self.buffer[self.oldest_item_idx+idx] for idx in indicies]
        return indicies, weights, samples

    def sample(self, sample_size):
        return self.random_sample_node(sample_size)

    def update(self, indexes, loss):
        # TODO make losses sequence of losses
        for idx in indexes:
            update(self.leaf_nodes[idx], (loss+OFFSET)**self.alpha)
        self.alpha = min(1, self.alpha+ALPHA_INCREMENT)
        self.beta = min(1, self.beta+BETA_INCREMENT)

    def log(self):
        pass
        # 3logging.info(f"max buffer size: {self.max_length}")

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
    
###############################################################################

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