import numpy as np
def meanSquaredError(prediction, expectation) -> np.array:
    """
    Returns the loss value, loss function used is mean squared error (this may change).
    """
    return np.mean((expectation - prediction) ** 2)