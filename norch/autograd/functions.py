import math

class AddBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    def backward(self, gradient):
        return [gradient, gradient]
    
class AddBroadcastedBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    def backward(self, gradient):
        x, y = self.input
        grad_x = self._reshape_gradient(gradient, x.shape)
        grad_y = self._reshape_gradient(gradient, y.shape)

        return [grad_x, grad_y]
    
    def _reshape_gradient(self, gradient, shape):
        # Reduce gradient dimensions to match the target shape dimensions
        while len(gradient.shape) > len(shape):
            gradient = gradient.sum(axis=0)

        # Sum along axes where the target shape dimension is 1
        for i in range(len(shape)):
            if shape[i] == 1:
                gradient = gradient.sum(axis=i)

        return gradient
    
class SubBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    def backward(self, gradient):
        return [gradient, -gradient]
    
class SubBroadcastedBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    def backward(self, gradient):
        x, y = self.input
        grad_x = self._reshape_gradient(gradient, x.shape)
        grad_y = self._reshape_gradient(gradient, y.shape)
        return [grad_x, -grad_y]
    
    def _reshape_gradient(self, gradient, shape):
        # Reduce gradient dimensions to match the target shape dimensions
        while len(gradient.shape) > len(shape):
            gradient = gradient.sum(axis=0)

        # Sum along axes where the target shape dimension is 1
        for i in range(len(shape)):
            if shape[i] == 1:
                gradient = gradient.sum(axis=i)
        return gradient
    
class ScalarMulBackward:
    def __init__(self, x, scalar):
        self.input = [x]
        self.scalar = scalar

    def backward(self, gradient):
        return [gradient * self.scalar]

class ElementwiseMulBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    def backward(self, gradient):
        x = self.input[0]
        y = self.input[1]
        return [y * gradient, x * gradient]
    
class MatmulBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    def backward(self, gradient):
        x, y = self.input
        
        if x.ndim != y.ndim: # broadcasted case
            aux = (gradient @ y.transpose(-1,-2))
            aux_sum = aux.sum(axis=0)
            return [aux_sum, x.transpose(-1,-2) @ gradient]
        else:
            return [gradient @ y.transpose(-1,-2), x.transpose(-1,-2) @ gradient]
        
"""class PowBackward:
    def __init__(self, x, power):
        self.input = [x]
        self.power = power

    def backward(self, gradient):
        return [(gradient * self.power) * (self.input[0]) ** (self.power - 1)]"""
    
class PowBackward:
    def __init__(self, base, exponent):
        self.input = [base, exponent]

    def backward(self, gradient):
        base, exponent = self.input[0], self.input[1]

        if isinstance(base, (int, float)):
            grad_base = gradient * (base ** (exponent - 1))
            grad_exponent = (gradient * base ** exponent) * math.log(base)

        else:
            grad_base = gradient * exponent * (base ** (exponent - 1))
            grad_exponent = (gradient * base ** exponent) * (base.log())

        return [grad_base, grad_exponent]

    
class LogBackward:
    def __init__(self, x):
        self.input = [x]

    def backward(self, gradient):

        grad_input = gradient / self.input[0]

        return [grad_input]
       
class SumBackward:
    def __init__(self, x, axis=None):
        self.input = [x]
        self.axis = axis

    def backward(self, gradient):
        input_shape = self.input[0].shape
        if self.axis == -1:
            # If axis is None, sum reduces the tensor to a scalar.
            grad_output = float(gradient.tensor.contents.data[0]) * self.input[0].ones_like()
        else:
            # Broadcast the gradient to the input shape along the specified axis.
            grad_output_shape = list(input_shape)
            grad_output_shape[self.axis] = 1
            grad_output = gradient.reshape(grad_output_shape)
            grad_output = grad_output + self.input[0].zeros_like()
        
        return [grad_output]
class ReshapeBackward:
    def __init__(self, x):
        self.input = [x]

    def backward(self, gradient):
        return [gradient.reshape(self.input[0].shape)]
    
class TransposeBackward:
    def __init__(self, x, axis1, axis2):
        self.input = [x]
        self.axis1 = axis1
        self.axis2 = axis2

    def backward(self, gradient):
        return [gradient.transpose(self.axis2, self.axis1)]
    
class TBackward:
    def __init__(self, x):
        self.input = [x]

    def backward(self, gradient):
        return [gradient.T]

    
class DivisionBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    def backward(self, gradient):
        x, y = self.input
        grad_x = gradient / y
        grad_y = -1 * gradient * (x / (y * y))
        return [grad_x, grad_y]
    
class SinBackward:
    def __init__(self, x):
        self.input = [x]

    def backward(self, gradient):
        x = self.input[0]
        return [gradient * x.cos()]
    
class CosBackward:
    def __init__(self, x):
        self.input = [x]

    def backward(self, gradient):
        x = self.input[0]
        return [-gradient * x.sin()]


