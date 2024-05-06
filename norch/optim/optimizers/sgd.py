from ..optimizer import Optimizer
from norch.tensor import Tensor

class SGD(Optimizer):
    def __init__(self, parameters, lr=1e-1, momentum=0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self._cache = {'velocity': [p.zeros_like() for (_, _, p) in self.parameters]}

    def step(self):
        for i, (module, name, parameter) in enumerate(self.parameters):
            velocity = self._cache['velocity'][i]

            velocity = self.momentum * velocity - self.lr * parameter.grad

            parameter += velocity

            setattr(module, name, parameter)

            self._cache['velocity'][i] = velocity
