import numpy as np
class Activation:
    def __init__(self, function, dFunction):
        self.function  = function
        self.dFunction = dFunction

    def forwardPropagation(self, input):
        self.input = input
        return self.function(self.input)
    
    def backPropagation(self, dOutput, learningRate):
        return np.multiply(dOutput, self.dFunction(self.input))