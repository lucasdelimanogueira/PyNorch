from norch.tensor import Tensor
from norch.utils import functions
import random

class Parameter(Tensor):
    """
    A parameter is a trainable tensor.
    """
    def __init__(self, shape):
        data = functions.generate_random_list(shape=shape)
        super().__init__(data, requires_grad=True)