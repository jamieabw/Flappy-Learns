import numpy as np

def meanSquaredError(prediction, expectation, action) -> float:
    """
    Compute MSE loss only on the chosen action.
    prediction, expectation: arrays of shape (num_actions,)
    action: int, index of the chosen action
    """
    diff = expectation[action] - prediction[action]
    return 0.5 * diff ** 2  # 0.5 factor common in MSE derivative simplification

def dMeanSquaredError(prediction, expectation, action) -> np.array:
    """
    Derivative of MSE wrt prediction for all actions:
    zero except chosen action where gradient is (prediction - expectation).
    """
    grad = np.zeros_like(prediction)
    grad[action] = prediction[action] - expectation[action]
    return grad

def huberLoss(prediction, expectation, action, delta=1.0) -> float:
    """
    Huber loss only on the chosen action.
    """
    error = prediction[action] - expectation[action]
    if abs(error) <= delta:
        return 0.5 * error ** 2
    else:
        return delta * (abs(error) - 0.5 * delta)

def dHuberLoss(prediction, expectation, action, delta=1.0) -> np.array:
    """
    Derivative of Huber loss wrt prediction.
    Zero except chosen action where derivative is:
      error if |error| <= delta
      delta * sign(error) otherwise
    """
    grad = np.zeros_like(prediction)
    error = prediction[action] - expectation[action]
    if abs(error) <= delta:
        grad[action] = error
    else:
        grad[action] = delta * np.sign(error)
    return grad
