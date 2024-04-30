class AddBackward:
    def __init__(self, x, y):
        self.tensors = [x, y]

    def backward(self, gradient):
        return [gradient, gradient]
    
class SubBackward:
    def __init__(self, x, y):
        self.tensors = [x, y]

    def backward(self, gradient):
        return [gradient, -gradient]
    
class ScalarMulBackward:
    def __init__(self, x, scalar):
        self.tensors = [x]
        self.scalar = scalar

    def backward(self, gradient):
        return [gradient * self.scalar]


class ElementwiseMulBackward:
    def __init__(self, x, y):
        self.tensors = [x, y]

    def backward(self, gradient):
        return [gradient * self.tensors[1], gradient * self.tensors[0]]
    
class SumBackward:
    def __init__(self, x):
        self.tensor = x

    def backward(self, gradient):
        # Since sum reduces a tensor to a scalar, gradient is broadcasted to match the original shape.
        return self.tensor.ones_like() * float(gradient.tensor.contents.data.value)


