from abc import ABC
from norch.tensor import Tensor

class Optimizer(ABC):
    """
    Abstract class for optimizers
    """

    def __init__(self, parameters):
        if isinstance(parameters, Tensor):
            raise TypeError("parameters should be an iterable but got {}".format(type(parameters)))
        elif isinstance(parameters, dict):
            parameters = parameters.values()

        self.parameters = list(parameters)

    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        for module, name, parameter in self.parameters:
            parameter.zero_grad()