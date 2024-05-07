import unittest
import norch
from norch import utils
import torch

class TestTensorOperations(unittest.TestCase):
    def test_creation_and_conversion(self):
        """
        Test creation and convertion of norch tensor to pytorch
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]])
        torch_tensor = utils.to_torch(norch_tensor)
        self.assertTrue(torch.is_tensor(torch_tensor))

    def test_addition(self):
        """
        Test addition two tensors: tensor1 + tensor2
        """
        norch_tensor1 = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]])
        norch_tensor2 = norch.Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
        norch_result = norch_tensor1 + norch_tensor2
        torch_result = utils.to_torch(norch_result)

        torch_tensor1 = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]])
        torch_tensor2 = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
        torch_expected = torch_tensor1 + torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_subtraction(self):
        """
        Test subtraction of two tensors: tensor1 - tensor2
        """
        norch_tensor1 = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]])
        norch_tensor2 = norch.Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
        norch_result = norch_tensor1 - norch_tensor2
        torch_result = utils.to_torch(norch_result)

        torch_tensor1 = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]])
        torch_tensor2 = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
        torch_expected = torch_tensor1 - torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_division_by_scalar(self):
        """
        Test division of a tensor by a scalar: tensor / scalar
        """
        norch_tensor = norch.Tensor([[[2, 4], [6, -8]], [[10, 12], [14, 16]]])
        scalar = 2
        norch_result = norch_tensor / scalar
        torch_result = utils.to_torch(norch_result)

        torch_tensor = torch.tensor([[[2, 4], [6, -8]], [[10, 12], [14, 16]]])
        torch_expected = torch_tensor / scalar

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_scalar_division_by_tensor(self):
        """
        Test scalar division by a tensor: scalar / tensor
        """
        scalar = 10
        norch_tensor = norch.Tensor([[[2, 4], [6, -8]], [[10, 12], [14, 16]]])
        norch_result = scalar / norch_tensor
        torch_result = utils.to_torch(norch_result)

        torch_tensor = torch.tensor([[[2, 4], [6, -8]], [[10, 12], [14, 16]]])
        torch_expected = scalar / torch_tensor

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_matrix_multiplication(self):
        """
        Test matrix multiplication: tensor1 @ tensor2
        """
        norch_tensor1 = norch.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        norch_tensor2 = norch.Tensor([[[1, 0], [0, 1]], [[-1, 0], [0, -1]]])
        norch_result = norch_tensor1 @ norch_tensor2
        torch_result = utils.to_torch(norch_result)

        torch_tensor1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        torch_tensor2 = torch.tensor([[[1, 0], [0, 1]], [[-1, 0], [0, -1]]])
        torch_expected = torch_tensor1 @ torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_elementwise_multiplication_by_scalar(self):
        """
        Test elementwise multiplication of a tensor by a scalar: tensor * scalar
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]])
        scalar = 2
        norch_result = norch_tensor * scalar
        torch_result = utils.to_torch(norch_result)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]])
        torch_expected = torch_tensor * scalar

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_elementwise_multiplication_by_tensor(self):
        """
        Test elementwise multiplication of two tensors: tensor1 * tensor2
        """
        norch_tensor1 = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]])
        norch_tensor2 = norch.Tensor([[[2, 2], [2, 2]], [[2, 2], [2, 2]]])
        norch_result = norch_tensor1 * norch_tensor2
        torch_result = utils.to_torch(norch_result)

        torch_tensor1 = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]])
        torch_tensor2 = torch.tensor([[[2, 2], [2, 2]], [[2, 2], [2, 2]]])
        torch_expected = torch_tensor1 * torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_reshape(self):
        """
        Test reshaping of a tensor: tensor.reshape(shape)
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]])
        new_shape = [2, 4]
        norch_result = norch_tensor.reshape(new_shape)
        torch_result = utils.to_torch(norch_result)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]])
        torch_expected = torch_tensor.reshape(new_shape)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_transpose(self):
        """
        Test transposition of a tensor: tensor.transpose(dim1, dim2)
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]])
        dim1, dim2 = 0, 2
        norch_result = norch_tensor.transpose(dim1, dim2)
        torch_result = utils.to_torch(norch_result)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]])
        torch_expected = torch_tensor.transpose(dim1, dim2)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_logarithm(self):
        """
        Test elementwise logarithm of a tensor: tensor.log()
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        norch_result = norch_tensor.log()
        torch_result = utils.to_torch(norch_result)

        torch_tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        torch_expected = torch.log(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_sum(self):
        """
        Test summation of a tensor: tensor.sum()
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]])
        norch_result = norch_tensor.sum()
        torch_result = utils.to_torch(norch_result)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]])
        torch_expected = torch.sum(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_transpose_T(self):
        """
        Test transposition of a tensor: tensor.T
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]])
        norch_result = norch_tensor.T
        torch_result = utils.to_torch(norch_result)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]])
        torch_expected = torch.transpose(torch_tensor, 0, 2)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))


if __name__ == '__main__':
    unittest.main()
