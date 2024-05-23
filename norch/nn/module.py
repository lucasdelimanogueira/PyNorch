from .parameter import Parameter
from collections import OrderedDict
from abc import ABC
import pickle
import json
import inspect
import warnings

class Module(ABC):
    """
    Abstract class for modules
    """
    def __init__(self):
        self._modules = OrderedDict()
        self._params = OrderedDict()
        self._grads = OrderedDict()
        self.training = True

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def train(self):
        self.training = True
        for param in self.parameters():
            param.requires_grad = True

    def eval(self):
        self.training = False
        for param in self.parameters():
            param.requires_grad = False

    def parameters(self):
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield self, name, value
            elif isinstance(value, Module):
                yield from value.parameters()

    def modules(self):
        yield from self._modules.values()

    def gradients(self):
        for module in self.modules():
            yield module._grads

    def zero_grad(self):
        for _, _, parameter in self.parameters():
            parameter.zero_grad()

    def to(self, device):
        for _, _, parameter in self.parameters():
            parameter.to(device)

        return self
    
    def state_dict(self):
        state = OrderedDict()
        for i, param in enumerate(self.parameters()):
            state[f'param{i}'] = param.tolist()
        return state
    
    def load_state(self, state_dict):
        for i, param in self.parameters():
            data = state_dict[f'param{i}']
            if param.shape != data.shape:
                warnings.warn(f"The 'state_dict' shape does not match model's parameter shape. "
                              f"Got {data.shape}, expected {param.shape}.")
            param.data = Parameter(data=data)

    def save(self, filename='model.pickle'):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def save_dict(self, filename='state_dict.json'):
        state = self.state_dict()
        with open(filename, 'w') as f:
            json.dump(state, f)

    def inner_repr(self):
        return ""

    def __repr__(self):
        string = f"{self.get_name()}("
        tab = "   "
        modules = self._modules
        if modules == {}:
            string += f'\n{tab}(parameters): {self.inner_repr()}'
        else:
            for key, module in modules.items():
                string += f"\n{tab}({key}): {module.get_name()}({module.inner_repr()})"
        return f'{string}\n)'
    
    def get_name(self):
        return self.__class__.__name__
    
    def __setattr__(self, key, value):
        self.__dict__[key] = value

        if isinstance(value, Module):
            self._modules[key] = value
        elif isinstance(value, Parameter):
            self._params[key] = value