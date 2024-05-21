import unittest
import norch
from norch.utils import utils_unittests as utils
import torch
import sys
import os

class TestTensorOperations(unittest.TestCase):

    def setUp(self):
        self.device = os.environ.get('device')
        if self.device is None or self.device != 'cuda':
            self.device = 'cpu'

        print(f"Running tests on: {self.device}")

    def test_creation_and_conversion(self):
        """
        Test creation and convertion of norch tensor to pytorch
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_tensor = utils.to_torch(norch_tensor)
        self.assertTrue(torch.is_tensor(torch_tensor))

    def test_addition(self):
        """
        Test addition two tensors: tensor1 + tensor2
        """
        norch_tensor1 = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_tensor2 = norch.Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).to(self.device)
        norch_result = norch_tensor1 + norch_tensor2
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor1 = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_tensor2 = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).to(self.device)
        torch_expected = torch_tensor1 + torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_addition_broadcasted(self):
        """
        Test addition of two tensors with broadcasting: tensor1 + tensor2
        """
        norch_tensor1 = norch.Tensor([[[1, 2, 3], [4, 5, 6]]]).to(self.device)  # Shape (1, 2, 3)
        norch_tensor2 = norch.Tensor([1, 1, 1]).to(self.device)  # Shape (3)
        norch_result = norch_tensor1 + norch_tensor2
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor1 = torch.tensor([[[1, 2, 3], [4, 5, 6]]]).to(self.device)  # Shape (1, 2, 3)
        torch_tensor2 = torch.tensor([1, 1, 1]).to(self.device)  # Shape (3)
        torch_expected = torch_tensor1 + torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        norch_tensor1 = norch.Tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]]).to(self.device)  # Shape (1, 2, 3)
        norch_tensor2 = norch.Tensor([[10, 10], [5, 6]]).to(self.device)  # Shape (3)
        norch_result = norch_tensor1 + norch_tensor2
        torch_result = utils.to_torch(norch_result).to(self.device)
        
        torch_tensor1 = torch.tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]]).to(self.device)  # Shape (1, 2, 3)
        torch_tensor2 = torch.tensor([[[10, 10], [5, 6]]]).to(self.device)  # Shape (3)
        torch_expected = torch_tensor1 + torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        # reversed order broadcasting
        norch_tensor1 = norch.Tensor([[0, 2]]).to(self.device) 
        norch_tensor2 = norch.Tensor([[3, 4], [5, -1]]).to(self.device) 
        norch_result = norch_tensor1 + norch_tensor2
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor1 = torch.tensor([[0, 2]]).to(self.device)  
        torch_tensor2 = torch.tensor([[3, 4], [5, -1]]).to(self.device) 
        torch_expected = torch_tensor1 + torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        norch_result = norch_tensor2 + norch_tensor1
        torch_expected = torch_tensor2 + torch_tensor1

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))



    def test_subtraction(self):
        """
        Test subtraction of two tensors: tensor1 - tensor2
        """
        norch_tensor1 = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_tensor2 = norch.Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).to(self.device)
        norch_result = norch_tensor1 - norch_tensor2
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor1 = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_tensor2 = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).to(self.device)
        torch_expected = torch_tensor1 - torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_broadcasting_subtraction(self):
        """
        Test subtraction of two tensors with broadcasting: tensor1 - tensor2
        """
        norch_tensor1 = norch.Tensor([[[1, 2, 3], [4, 5, 6]]]).to(self.device)  # Shape (1, 2, 3)
        norch_tensor2 = norch.Tensor([1, 1, 1]).to(self.device)  # Shape (3)
        norch_result = norch_tensor1 - norch_tensor2
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor1 = torch.tensor([[[1, 2, 3], [4, 5, 6]]]).to(self.device)  # Shape (1, 2, 3)
        torch_tensor2 = torch.tensor([1, 1, 1]).to(self.device)  # Shape (3)
        torch_expected = torch_tensor1 - torch_tensor2 

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        # reversed order broadcasting
        norch_result = norch_tensor2 - norch_tensor1
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_expected = torch_tensor2 - torch_tensor1

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))


    def test_division_by_scalar(self):
        """
        Test division of a tensor by a scalar: tensor / scalar
        """
        norch_tensor = norch.Tensor([[[2, 4], [6, -8]], [[10, 12], [14, 16]]]).to(self.device)
        scalar = 2
        norch_result = norch_tensor / scalar
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[2, 4], [6, -8]], [[10, 12], [14, 16]]]).to(self.device)
        torch_expected = torch_tensor / scalar

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_scalar_division_by_tensor(self):
        """
        Test scalar division by a tensor: scalar / tensor
        """
        scalar = 10
        norch_tensor = norch.Tensor([[[2, 4], [6, -8]], [[10, 12], [14, 16]]]).to(self.device)
        norch_result = scalar / norch_tensor
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[2, 4], [6, -8]], [[10, 12], [14, 16]]]).to(self.device)
        torch_expected = scalar / torch_tensor

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_matrix_multiplication(self):
        """
        Test matrix multiplication: tensor1 @ tensor2
        """
        norch_tensor1 = norch.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_tensor2 = norch.Tensor([[[1, 0], [0, 1]], [[-1, 0], [0, -1]]]).to(self.device)
        norch_result = norch_tensor1 @ norch_tensor2
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_tensor2 = torch.tensor([[[1, 0], [0, 1]], [[-1, 0], [0, -1]]]).to(self.device)
        torch_expected = torch_tensor1 @ torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_elementwise_multiplication_by_scalar(self):
        """
        Test elementwise multiplication of a tensor by a scalar: tensor * scalar
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        scalar = 2
        norch_result = norch_tensor * scalar
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch_tensor * scalar

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_elementwise_multiplication_by_tensor(self):
        """
        Test elementwise multiplication of two tensors: tensor1 * tensor2
        """
        norch_tensor1 = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_tensor2 = norch.Tensor([[[2, 2], [2, 2]], [[2, 2], [2, 2]]]).to(self.device)
        norch_result = norch_tensor1 * norch_tensor2
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor1 = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_tensor2 = torch.tensor([[[2, 2], [2, 2]], [[2, 2], [2, 2]]]).to(self.device)
        torch_expected = torch_tensor1 * torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_reshape(self):
        """
        Test reshaping of a tensor: tensor.reshape(shape)
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        new_shape = [2, 4]
        norch_result = norch_tensor.reshape(new_shape)
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch_tensor.reshape(new_shape)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_unsqueeze(self):
        """
        Test unsqueeze operation on a tensor
        """
        norch_tensor = norch.Tensor([[1, 2], [3, 4]]).to(self.device)
        
        # Unsqueeze at dim=0
        norch_unsqueeze_0 = norch_tensor.unsqueeze(0)
        torch_unsqueeze_0 = utils.to_torch(norch_unsqueeze_0).to(self.device)
        torch_tensor = torch.tensor([[1, 2], [3, 4]]).to(self.device)
        torch_expected_0 = torch_tensor.unsqueeze(0)
        self.assertTrue(utils.compare_torch(torch_unsqueeze_0, torch_expected_0))

        # Unsqueeze at dim=1
        norch_unsqueeze_1 = norch_tensor.unsqueeze(1)
        torch_unsqueeze_1 = utils.to_torch(norch_unsqueeze_1).to(self.device)
        torch_expected_1 = torch_tensor.unsqueeze(1)
        self.assertTrue(utils.compare_torch(torch_unsqueeze_1, torch_expected_1))

        # Unsqueeze at dim=2
        norch_unsqueeze_2 = norch_tensor.unsqueeze(2)
        torch_unsqueeze_2 = utils.to_torch(norch_unsqueeze_2).to(self.device)
        torch_expected_2 = torch_tensor.unsqueeze(2)
        self.assertTrue(utils.compare_torch(torch_unsqueeze_2, torch_expected_2))

        # Unsqueeze at dim=-1
        norch_unsqueeze_neg_1 = norch_tensor.unsqueeze(-1)
        torch_unsqueeze_neg_1 = utils.to_torch(norch_unsqueeze_neg_1).to(self.device)
        torch_expected_neg_1 = torch_tensor.unsqueeze(-1)
        self.assertTrue(utils.compare_torch(torch_unsqueeze_neg_1, torch_expected_neg_1))

        # Unsqueeze at dim=-2
        norch_unsqueeze_neg_2 = norch_tensor.unsqueeze(-2)
        torch_unsqueeze_neg_2 = utils.to_torch(norch_unsqueeze_neg_2).to(self.device)
        torch_expected_neg_2 = torch_tensor.unsqueeze(-2)
        self.assertTrue(utils.compare_torch(torch_unsqueeze_neg_2, torch_expected_neg_2))

    def test_transpose(self):
        """
        Test transposition of a tensor: tensor.transpose(dim1, dim2)
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        dim1, dim2 = 0, 2
        norch_result = norch_tensor.transpose(dim1, dim2)
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch_tensor.transpose(dim1, dim2)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_logarithm(self):
        """
        Test elementwise logarithm of a tensor: tensor.log()
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_result = norch_tensor.log()
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch.log(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_sum(self):
        """
        Test summation of a tensor: tensor.sum()
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_result = norch_tensor.sum()
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch.sum(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_sum_axis(self):
        """
        Test summation of a tensor along a specific axis without keeping the dimensions
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_result = norch_tensor.sum(axis=1)
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch.sum(torch_tensor, dim=1)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        # negative axis

        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_result = norch_tensor.sum(axis=-2)
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch.sum(torch_tensor, dim=-2)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))


    def test_sum_axis_keepdim(self):
        """
        Test summation of a tensor along a specific axis with keepdim=True
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_result = norch_tensor.sum(axis=1, keepdim=True)
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch.sum(torch_tensor, dim=1, keepdim=True)
        
        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_max(self):
        """
        Test max of a tensor: tensor.max()
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_result = norch_tensor.max()
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch.max(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_max_axis(self):
        """
        Test max of a tensor along a specific axis without keeping the dimensions
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_result = norch_tensor.max(axis=1)
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected, _ = torch.max(torch_tensor, dim=1)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        # negative axis

        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_result = norch_tensor.max(axis=-1)
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected, _ = torch.max(torch_tensor, dim=-1)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))


    def test_max_axis_keepdim(self):
        """
        Test max of a tensor along a specific axis with keepdim=True
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_result = norch_tensor.max(axis=1, keepdim=True)
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected, _ = torch.max(torch_tensor, dim=1, keepdim=True)
        
        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_min(self):
        """
        Test min of a tensor: tensor.min()
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_result = norch_tensor.min()
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch.min(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_min_axis(self):
        """
        Test min of a tensor along a specific axis without keeping the dimensions
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_result = norch_tensor.min(axis=1)
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected, _ = torch.min(torch_tensor, dim=1)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        # negative axis

        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_result = norch_tensor.min(axis=-1)
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected, _ = torch.min(torch_tensor, dim=-1)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))


    def test_min_axis_keepdim(self):
        """
        Test min of a tensor along a specific axis with keepdim=True
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_result = norch_tensor.min(axis=1, keepdim=True)
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected, _ = torch.min(torch_tensor, dim=1, keepdim=True)
        
        self.assertTrue(utils.compare_torch(torch_result, torch_expected))


    def test_transpose_T(self):
        """
        Test transposition of a tensor: tensor.T
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_result = norch_tensor.T
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch.transpose(torch_tensor, 0, 2)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_matmul(self):
        """
        Test matrix multiplication: MxP = NxM @ MxP
        """
        # Creating batched tensors for Norch
        norch_tensor1 = norch.Tensor([[1, 2], [3, -4], [5, 6], [7, 8]]).to(self.device)
        norch_tensor2 = norch.Tensor([[2, 3, 1, 0, 4], [5, -1, 2, 3, 0]]).to(self.device)

        norch_result = norch_tensor1 @ norch_tensor2
        torch_result = utils.to_torch(norch_result).to(self.device)

        # Converting to PyTorch tensors for comparison
        torch_tensor1 = torch.tensor([[1, 2], [3, -4], [5, 6], [7, 8]]).to(self.device)
        torch_tensor2 = torch.tensor([[2, 3, 1, 0, 4], [5, -1, 2, 3, 0]]).to(self.device)

        torch_expected = torch.matmul(torch_tensor1, torch_tensor2)

        # Comparing results
        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_reshape_then_matmul(self):
        """
        Test reshaping a tensor followed by matrix multiplication: (tensor.reshape(shape) @ other_tensor)
        """
        norch_tensor = norch.Tensor([[1, 2], [3, -4], [5, 6], [7, 8]]).to(self.device)
        new_shape = [2, 4]
        norch_reshaped = norch_tensor.reshape(new_shape)
        
        norch_result = norch_reshaped @ norch_tensor
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[1, 2], [3, -4], [5, 6], [7, 8]]).to(self.device)
        torch_expected = torch_tensor.reshape(new_shape) @ torch_tensor

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_batched_matmul(self):
        """
        Test batched matrix multiplication: BxMxP = BxNxM @ BxMxP
        """
        B = 3  # Batch size

        # Creating batched tensors for Norch
        norch_tensor1 = norch.Tensor([[[1, 2], [3, -4], [5, 6], [7, 8]] for _ in range(B)]).to(self.device)
        norch_tensor2 = norch.Tensor([[[2, 3, 1, 0, 4], [5, -1, 2, 3, 0]] for _ in range(B)]).to(self.device)

        norch_result = norch_tensor1 @ norch_tensor2
        torch_result = utils.to_torch(norch_result).to(self.device)

        # Converting to PyTorch tensors for comparison
        torch_tensor1 = torch.tensor([[[1, 2], [3, -4], [5, 6], [7, 8]] for _ in range(B)]).to(self.device)
        torch_tensor2 = torch.tensor([[[2, 3, 1, 0, 4], [5, -1, 2, 3, 0]] for _ in range(B)]).to(self.device)

        torch_expected = torch.matmul(torch_tensor1, torch_tensor2)

        # Comparing results
        self.assertTrue(utils.compare_torch(torch_result, torch_expected))


    def test_broadcasted_batched_matmul(self):
        """
        Test broadcasted batched matrix multiplication: BxMxP = NxM @ BxMxP
        """
        B = 3  # Batch size

        # Creating batched tensors for Norch
        norch_tensor1 = norch.Tensor([[1, 2], [3, -4], [5, 6], [7, 8]]).to(self.device)
        norch_tensor2 = norch.Tensor([[[2, 3, 1, 0, 4], [5, -1, 2, 3, 0]] for _ in range(B)]).to(self.device)

        norch_result = norch_tensor1 @ norch_tensor2
        torch_result = utils.to_torch(norch_result).to(self.device)

        # Converting to PyTorch tensors for comparison
        torch_tensor1 = torch.tensor([[1, 2], [3, -4], [5, 6], [7, 8]]).to(self.device)
        torch_tensor2 = torch.tensor([[[2, 3, 1, 0, 4], [5, -1, 2, 3, 0]] for _ in range(B)]).to(self.device)

        torch_expected = torch.matmul(torch_tensor1, torch_tensor2)

        # Comparing results
        self.assertTrue(utils.compare_torch(torch_result, torch_expected))



    def test_transpose_then_matmul(self):
        """
        Test transposing a tensor followed by matrix multiplication: (tensor.transpose(dim1, dim2) @ other_tensor)
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).to(self.device)
        dim1, dim2 = 0, 2
        norch_result = norch_tensor.transpose(dim1, dim2) @ norch_tensor
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch_tensor.transpose(dim1, dim2) @ torch_tensor

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_add_div_matmul_then_reshape(self):
        """
        Test a combination of operations: (tensor.sum() + other_tensor) / scalar @ another_tensor followed by reshape
        """
        norch_tensor1 = norch.Tensor([[[1., 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_tensor2 = norch.Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).to(self.device)
        scalar = 2
        new_shape = [2, 4]
        norch_result = ((norch_tensor1 + norch_tensor2) / scalar) @ norch_tensor1
        norch_result = norch_result.reshape(new_shape)
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor1 = torch.tensor([[[1., 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_tensor2 = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).to(self.device)
        torch_expected = ((torch_tensor1 + torch_tensor2) / scalar) @ torch_tensor1
        torch_expected = torch_expected.reshape(new_shape)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_scalar_power_tensor(self):
        """
        Test scalar power of a tensor: scalar ** tensor
        """
        scalar = 3
        norch_tensor = norch.Tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_result = scalar ** norch_tensor
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = scalar ** torch_tensor

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_tensor_power_scalar(self):
        """
        Test tensor power of a scalar: tensor ** scalar
        """
        scalar = 3
        norch_tensor = norch.Tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_result = norch_tensor ** scalar
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch_tensor ** scalar

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_tensor_sin(self):
        """
        Test sine function on tensor
        """
        norch_tensor = norch.Tensor([[[0, 30], [45, 60]], [[90, 120], [135, 180]]]).to(self.device)
        norch_result = norch_tensor.sin()
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[0, 30], [45, 60]], [[90, 120], [135, 180]]]).to(self.device)
        torch_expected = torch.sin(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_tensor_cos(self):
        """
        Test cosine function on tensor
        """
        norch_tensor = norch.Tensor([[[0, 30], [45, 60]], [[90, 120], [135, 180]]]).to(self.device)
        norch_result = norch_tensor.cos()
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor = torch.tensor([[[0, 30], [45, 60]], [[90, 120], [135, 180]]]).to(self.device)
        torch_expected = torch.cos(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_equal(self):
        """
        Test equal two tensors: tensor1.equal(tensor2)
        """
        norch_tensor1 = norch.Tensor([[[1, 2], [3, -4]], [[5, 1], [7, 8]]]).to(self.device)
        norch_tensor2 = norch.Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).to(self.device)
        norch_result = norch_tensor1.equal(norch_tensor2)
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor1 = torch.tensor([[[1, 2], [3, -4]], [[5, 1], [7, 8]]]).to(self.device)
        torch_tensor2 = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).to(self.device)
        torch_expected = (torch_tensor1 == torch_tensor2).float()

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_broadcasted_equal(self):
        """
        Test broadcasted equal two tensors: tensor1.equal(tensor2)
        """
        norch_tensor1 = norch.Tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]]).to(self.device)
        norch_tensor2 = norch.Tensor([[[10, 10]], [[5, 6]]]).to(self.device)
        norch_result = norch_tensor1.equal(norch_tensor2)
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor1 = torch.tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]]).to(self.device)
        torch_tensor2 = torch.tensor([[[10, 10]], [[5, 6]]]).to(self.device)
        torch_expected = (torch_tensor1 == torch_tensor2).float()

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        norch_tensor1 = norch.Tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]]).to(self.device)
        norch_tensor2 = norch.Tensor([[[10.0,], [-4.0,]],[[6.0,], [8.0,]]]).to(self.device)
        norch_result = norch_tensor1.equal(norch_tensor2)
        torch_result = utils.to_torch(norch_result).to(self.device)

        torch_tensor1 = torch.tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]]).to(self.device)
        torch_tensor2 = torch.tensor([[[10.0,], [-4.0,]],[[6.0,], [8.0,]]]).to(self.device)
        torch_expected = (torch_tensor1 == torch_tensor2).float()

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))
    

    def test_zeros_like(self):
        """
        Test creating a tensor of zeros with the same shape as another tensor.
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_zeros = norch_tensor.zeros_like()
        torch_zeros_result = utils.to_torch(norch_zeros).to(self.device)

        torch_tensor_expected = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_zeros_expected = torch.zeros_like(torch_tensor_expected)

        self.assertTrue(utils.compare_torch(torch_zeros_result, torch_zeros_expected))

    def test_ones_like(self):
        """
        Test creating a tensor of ones with the same shape as another tensor.
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        norch_ones = norch_tensor.ones_like()
        torch_ones_result = utils.to_torch(norch_ones).to(self.device)

        torch_tensor_expected = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_ones_expected = torch.ones_like(torch_tensor_expected)

        self.assertTrue(utils.compare_torch(torch_ones_result, torch_ones_expected))


if __name__ == '__main__':
    unittest.main()
