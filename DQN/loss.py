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

def huberLoss(prediction, expectation, delta=1.0) -> np.array:
    """
    Returns the Huber loss.
    """
    error = prediction - expectation
    isSmall = np.abs(error) <= delta
    squaredLoss = 0.5 * error ** 2
    linearLoss = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(isSmall, squaredLoss, linearLoss))

def dHuberLoss(prediction, expectation, delta=1.0) -> np.array:
    """
    Returns the derivative of Huber loss.
    """
    error = prediction - expectation
    grad = np.where(np.abs(error) <= delta, error, delta * np.sign(error))
    return grad
