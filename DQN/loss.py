import numpy as np
def meanSquaredError(prediction, expectation) -> np.array:
    """
    Returns the loss value, loss function used is mean squared error (this may change).
    """
    return np.mean((expectation - prediction) ** 2)
def dMeanSquaredError(prediction, expectation)-> np.array:
    """
    Returns the derivative value of the loss function used in back prop
    """
    n = len(prediction)
    return (2/n) * (prediction - expectation)