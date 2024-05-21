import math

def sigmoid(x):
    return 1.0 / (1.0 + (math.e) ** (-x)) 

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