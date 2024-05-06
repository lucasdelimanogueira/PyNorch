from .module import Module
import math

class Activation(Module):
    """
    Abstract classes for activations
    """
    def __init__(self):
        super(Activation, self).__init__()

    def forward(self, x):
        raise NotImplementedError
    

class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1.0 / (1.0 + (math.e) ** (-x)) 