import math

class AddBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    def backward(self, gradient):
        return [gradient, gradient]
    
class SubBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    def backward(self, gradient):
        return [gradient, -gradient]
    
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
    def __init__(self, x):
        self.input = [x]

    def backward(self, gradient):
        # Since sum reduces a tensor to a scalar, gradient is broadcasted to match the original shape.
        return [float(gradient.tensor.contents.data[0]) * self.input[0].ones_like()]
    
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


