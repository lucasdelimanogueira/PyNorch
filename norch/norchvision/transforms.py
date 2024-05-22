import norch

class ToTensor:
    def __call__(self, x):
        return norch.Tensor(x)

class Reshape:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, x):
        return x.reshape(self.shape)
    
class Sequential:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

