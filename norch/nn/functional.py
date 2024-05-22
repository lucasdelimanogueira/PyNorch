import math
import norch
import numpy as np
from norch.autograd.functions import *

def sigmoid(x):
    z = x.sigmoid()
    return z

def softmax(x, dim=None):
    if dim is not None and dim < 0:
        dim = x.ndim + dim
            
    x_max = x.max(axis=dim, keepdim=True)
    exp_x = math.e ** (x - x_max)

    if dim is not None:
        sum_exp_x = exp_x.sum(axis=dim, keepdim=True) + exp_x.zeros_like()
        return exp_x / sum_exp_x
    else:
        sum_exp_x = exp_x.sum()
        return exp_x / sum_exp_x
    
def one_hot_encode(x, num_classes):
    one_hot = [[0] * num_classes for _ in range(x.numel)]
    
    # Set the appropriate elements to 1
    for i in range(x.numel):
        target_idx = int(x.tensor.contents.data[i])
        one_hot[i][target_idx] = 1

    return norch.Tensor(one_hot)