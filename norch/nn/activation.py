from .module import Module
from . import functional as F
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
        return F.sigmoid(x)
    
class Softmax(Activation):
    def __init__(self, dim):
        super(Softmax, self).__init__()

        self.dim = dim

    def forward(self, x):
        return F.softmax(x, self.dim)