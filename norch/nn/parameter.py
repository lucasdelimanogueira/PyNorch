from tensor import Tensor
import random

class Parameter(Tensor):
    """
    A parameter is a trainable tensor.
    """
    def __init__(self, shape):
        data = []
        for dim_size in reversed(shape):
            random_dim = [random.random() for _ in range(dim_size)]
            data.insert(0, random_dim)

        super().__init__(data, requires_grad=True)