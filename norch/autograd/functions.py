class AddBackward:
    def __init__(self, x, y):
        self.tensors = [x, y]

    def backward(self, gradient):
        return [gradient, gradient]
