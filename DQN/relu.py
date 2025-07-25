from DQN.activation import Activation
import numpy as np
class Relu(Activation):
    """
    RELU activation, if <= 0 the value will be automatically zero
    otherwise, original value used.
    """
    def __init__(self):
        function = lambda x : np.maximum(0,x)
        dFunction = lambda x : (x > 0).astype(float)
        super().__init__(function, dFunction)
