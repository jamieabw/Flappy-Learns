from DQN.activation import Activation
import numpy as np

class LeakyRelu(Activation):
    def __init__(self, alpha=0.03):
        """
        Leaky ReLU activation function:
            f(x) = x if x > 0 else alpha * x
            f'(x) = 1 if x > 0 else alpha
        """
        function = lambda x: np.where(x > 0, x, alpha * x)
        dFunction = lambda x: np.where(x > 0, 1.0, alpha)
        super().__init__(function, dFunction)
