import numpy as np
from DQN.layer import Layer
class DenseLayer(Layer):
    def __init__(self, inputLength, outputLength):
        self.weights = np.random.randn(outputLength, inputLength)
        self.biases = np.random.randn(outputLength)

    def forwardPropagation(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.biases
    
    def backPropagation(self, output):
        return super().backPropagation(output)