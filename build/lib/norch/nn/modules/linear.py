from ..module import Module
from ..parameter import Parameter

class Linear(Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Parameter(shape=[self.output_dim, self.input_dim])

        if bias:
            self.bias = Parameter(shape=[self.output_dim, 1])
        else:
            self.bias = None

    def forward(self, x):
        if self.bias:
            z = self.weight @ x + self.bias
        else:
            z = self.weight @ x

        return z

    def inner_repr(self):
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, " \
               f"bias={True if self.bias is not None else False}"