from .module import *
import norch.distributed as dist
import os

class DistributedDataParallel(Module):
    def __init__(self, module):
        super().__init__()
        
        self.module = module

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def backward(self):
        self.module.backward()
        for module, name, parameter in self.parameters():
            dist.allreduce_sum_tensor(parameter)

        

    
