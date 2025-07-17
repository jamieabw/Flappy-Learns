import numpy as np
class Activation:
    def __init__(self, function, dFunction):
        self.function  = function
        self.dFunction = dFunction

    def forwardPropagation(self, input):
        self.input = input
        return self.function(self.input)
    
    def backPropagation(self, dOutput, learningRate=0.1):
        return np.multiply(dOutput, self.dFunction(self.input))