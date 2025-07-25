import numpy as np

DELTA = 1e-5 # small constant summed to all priorities so none are 0

class PriorityReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []

    def add(self, experience):
        """
        Adds a new experience to replay buffer and initially gives highest probability so guaranteed for sample
        """
        self.buffer.append(experience)
        self.priorities.append(1.0)
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]
            self.priorities = self.priorities[-self.capacity:] # this likely isnt the most efficient way of doing this, at all

    def sample(self, batchSize, beta=0.4) -> tuple:
        """
        Selects a batch of experiences to be used in training, higher TD error -> more likely to be chosen
        """
        scaledProbs = np.array(self.priorities) ** self.alpha
        probs = scaledProbs / scaledProbs.sum()
        indices = np.random.choice(len(self.buffer), batchSize, p=probs)
        samples = [self.buffer[i] for i in indices]
        n = len(self.buffer)
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights
    
    def updatePriorities(self, indices, newPriorities):
        """
        Sets true priorities for the experience, based on their actual calculated TD error.
        """
        for i, p in zip(indices, newPriorities):
            self.priorities[i] = abs(p) + DELTA