from .module import *
import norch.distributed as dist
import os
import norch

class DistributedDataParallel(Module):
    def __init__(self, module):
        super().__init__()
        
        self.module = module

        
        self.broadcast_parameters()
        self.register_grads_hooks()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def broadcast_parameters(self):
        """
        Broadcast parameters of device 0 to all devices
        """
        for _, _, parameter in self.parameters():
            dist.broadcast_tensor(parameter)
    
    def allreduce_grads_hook(grad):
        """
        Everytime a gradient is assign to some value, it calculates mean of this gradient among all devices
        """
        if isinstance(grad, norch.Tensor):
            dist.allreduce_sum_tensor(grad) 
            grad /= dist.get_world_size()
        return grad
    
    def register_grads_hooks(self):
        """
        Everytime a gradient is assign it calls this allreduce hook
        """
        for _, _, parameter in self.parameters():
            parameter.register_hook(self.allreduce_grads_hook)

    

    
    

        

    
