import numpy as np
class Activation:
    """
    Base class for activation functions.
    """
    def __init__(self, function, dFunction):
        self.function  = function
        self.dFunction = dFunction

    def forwardPropagation(self, input):
        self.input = input
        return self.function(self.input)
    
    def backPropagation(self, dOutput, learningRate=1e-5):
        return np.multiply(dOutput, self.dFunction(self.input))