import norch
from norch.utils import utils_unittests as utils

device = "cpu"

norch_tensor = norch.Tensor([[[[1, 2], [3, -4]], [[5, 6], [7, 8]]], [[[1, 2], [3, -4]], [[5, 6], [7, 8]]]]).to(device)
norch_result = norch_tensor.sum(axis=0)
