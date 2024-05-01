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
        return [gradient * self.input[1], gradient * self.input[0]]
    
class SumBackward:
    def __init__(self, x):
        self.input = [x]

    def backward(self, gradient):
        # Since sum reduces a tensor to a scalar, gradient is broadcasted to match the original shape.
        return [float(gradient.tensor.contents.data[0]) * self.input[0].ones_like()]


