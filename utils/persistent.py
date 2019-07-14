import collections
import random
from collections import deque


class Memory:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, sample):
        self.buffer.append(sample)

    def sample(self, batch_size):
        n = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, n)

    def size(self):
        return len(self.buffer)


class PGMemory:

    def __init__(self):
        self.buffer = []
        self.transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(self.transition(state=state, action=action, reward=reward, next_state=next_state, done=done))

    def clear(self):
        self.buffer.clear()

    def size(self):
        return len(self.buffer)
