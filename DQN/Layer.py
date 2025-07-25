class Layer:
    """
    Base class for layers
    """
    def __init__(self):
        self.weights = None
        self.biases = None

    def forwardPropagation(self, inputs):
        pass

    def backPropagation(self, output):
        pass