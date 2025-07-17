from DQN.activation import Activation
import numpy as np
class Relu(Activation):
    def __init__(self):
        function = lambda x : np.maximum(0,x)
        dFunction = lambda x : (x >= 0).astype(float)
        super().__init__(function, dFunction)
