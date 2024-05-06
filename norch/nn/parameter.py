from norch.tensor import Tensor
from norch.utils import utils
import random

class Parameter(Tensor):
    """
    A parameter is a trainable tensor.
    """
    def __init__(self, shape):
        data = utils.generate_random_list(shape=shape)
        super().__init__(data, requires_grad=True)