import unittest
import norch
from norch.utils import utils_unittests as utils
import torch
import os

class TestTensorAutograd(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get('device')
        if self.device is None or self.device != 'cuda':
            self.device = 'cpu'
        
        print(f"Running tests on: {self.device}")

        
    def test_addition(self):
        """
        Test autograd from addition two tensors: tensor1 + tensor2
        """
        norch_tensor1 = norch.Tensor([[[1, 2.5], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        norch_tensor2 = norch.Tensor([[[1, 1.], [1, 1.9]], [[1, 1], [1, 1]]], requires_grad=True).to(self.device)
        norch_result = (norch_tensor1 + norch_tensor2).sum()
        norch_result.backward()
        norch_tensor1_grad = utils.to_torch(norch_tensor1.grad).to(self.device)
        norch_tensor2_grad = utils.to_torch(norch_tensor2.grad).to(self.device)

        torch_tensor1 = torch.tensor([[[1, 2.5], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        torch_tensor2 = torch.tensor([[[1, 1.], [1, 1.9]], [[1, 1], [1, 1]]], requires_grad=True).to(self.device)
        torch_result = (torch_tensor1 + torch_tensor2).sum()
        torch_result.backward()
        torch_tensor1_grad = torch_tensor1.grad
        torch_tensor2_grad = torch_tensor2.grad

        self.assertTrue(utils.compare_torch(norch_tensor1_grad, torch_tensor1_grad))
        self.assertTrue(utils.compare_torch(norch_tensor2_grad, torch_tensor2_grad))

    def test_sum_axis(self):
        """
        Test autograd from sum specifying axis
        """
        norch_tensor1 = norch.Tensor([[[1, 2.5], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        norch_tensor2 = norch.Tensor([[[1, 1.], [1, 1.9]], [[1, 1], [1, 1]]], requires_grad=True).to(self.device)
        norch_result = (norch_tensor1 + norch_tensor2).sum(axis=0).sum(axis=0).sum()

        norch_result.backward()
        norch_tensor1_grad = utils.to_torch(norch_tensor1.grad).to(self.device)
        norch_tensor2_grad = utils.to_torch(norch_tensor2.grad).to(self.device)

        torch_tensor1 = torch.tensor([[[1, 2.5], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        torch_tensor2 = torch.tensor([[[1, 1.], [1, 1.9]], [[1, 1], [1, 1]]], requires_grad=True).to(self.device)
        
        torch_result = (torch_tensor1 + torch_tensor2).sum(axis=0).sum(axis=0).sum()
        torch_result.backward()
        torch_tensor1_grad = torch_tensor1.grad
        torch_tensor2_grad = torch_tensor2.grad
        
        self.assertTrue(utils.compare_torch(norch_tensor1_grad, torch_tensor1_grad))
        self.assertTrue(utils.compare_torch(norch_tensor2_grad, torch_tensor2_grad))

        norch_tensor1 = norch.Tensor([[[1, 2.5], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        norch_tensor2 = norch.Tensor([[[1, 1.], [1, 1.9]], [[1, 1], [1, 1]]], requires_grad=True).to(self.device)
        norch_result = (norch_tensor1 + norch_tensor2).sum(axis=1).sum()

        norch_result.backward()
        norch_tensor1_grad = utils.to_torch(norch_tensor1.grad).to(self.device)
        norch_tensor2_grad = utils.to_torch(norch_tensor2.grad).to(self.device)

        torch_tensor1 = torch.tensor([[[1, 2.5], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        torch_tensor2 = torch.tensor([[[1, 1.], [1, 1.9]], [[1, 1], [1, 1]]], requires_grad=True).to(self.device)
        
        torch_result = (torch_tensor1 + torch_tensor2).sum(axis=1).sum()
        torch_result.backward()
        torch_tensor1_grad = torch_tensor1.grad
        torch_tensor2_grad = torch_tensor2.grad
        
        self.assertTrue(utils.compare_torch(norch_tensor1_grad, torch_tensor1_grad))
        self.assertTrue(utils.compare_torch(norch_tensor2_grad, torch_tensor2_grad))
    
    
    def test_max(self):
        """
        Test autograd from max
        """
        norch_tensor = norch.Tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]], requires_grad=True).to(self.device)
        norch_result = norch_tensor.max()

        norch_result.backward()
        norch_tensor_grad = utils.to_torch(norch_tensor.grad).to(self.device)

        torch_tensor = torch.tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]], requires_grad=True).to(self.device)
        
        torch_result = torch_tensor.max()
        torch_result.backward()
        torch_tensor_grad = torch_tensor.grad
        
        self.assertTrue(utils.compare_torch(norch_tensor_grad, torch_tensor_grad))
    
    def test_max_axis(self):
        """
        Test autograd from max specifying axis
        """
        norch_tensor = norch.Tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]], requires_grad=True).to(self.device)
        norch_max_axis = norch_tensor.max(axis=1)
        norch_result = norch_max_axis.sum()

        norch_result.backward()
        norch_tensor_grad = utils.to_torch(norch_tensor.grad).to(self.device)

        torch_tensor = torch.tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]], requires_grad=True).to(self.device)

        torch_max_axis, _ = torch_tensor.max(axis=1)
        torch_result = torch_max_axis.sum()
        torch_result.backward()
        torch_tensor_grad = torch_tensor.grad
        
        self.assertTrue(utils.compare_torch(norch_tensor_grad, torch_tensor_grad))


        norch_tensor = norch.Tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]], requires_grad=True).to(self.device)
        norch_max_axis = norch_tensor.max(axis=2)
        norch_result = norch_max_axis.sum()

        norch_result.backward()
        norch_tensor_grad = utils.to_torch(norch_tensor.grad).to(self.device)

        torch_tensor = torch.tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]], requires_grad=True).to(self.device)

        torch_max_axis, _ = torch_tensor.max(axis=2)
        torch_result = torch_max_axis.sum()
        torch_result.backward()
        torch_tensor_grad = torch_tensor.grad
        
        self.assertTrue(utils.compare_torch(norch_tensor_grad, torch_tensor_grad))

        print(norch_tensor_grad, '\n\n', torch_tensor_grad)

    
    def test_min(self):
        """
        Test autograd from min
        """
        norch_tensor = norch.Tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]], requires_grad=True).to(self.device)
        norch_result = norch_tensor.min()

        norch_result.backward()
        norch_tensor_grad = utils.to_torch(norch_tensor.grad).to(self.device)

        torch_tensor = torch.tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]], requires_grad=True).to(self.device)
        
        torch_result = torch_tensor.min()
        torch_result.backward()
        torch_tensor_grad = torch_tensor.grad
        
        self.assertTrue(utils.compare_torch(norch_tensor_grad, torch_tensor_grad))

    def test_min_axis(self):
        """
        Test autograd from min specifying axis
        """
        norch_tensor = norch.Tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]], requires_grad=True).to(self.device)
        norch_min = norch_tensor.min(axis=1)
        norch_result = norch_min.sum()

        norch_result.backward()
        norch_tensor_grad = utils.to_torch(norch_tensor.grad).to(self.device)

        torch_tensor = torch.tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]], requires_grad=True).to(self.device)
        
        torch_min = torch_tensor.min(axis=1)
        torch_result = torch_min.sum()
        torch_result.backward()
        torch_tensor_grad = torch_tensor.grad
        
        self.assertTrue(utils.compare_torch(norch_tensor_grad, torch_tensor_grad))

        norch_tensor = norch.Tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]], requires_grad=True).to(self.device)
        norch_min = norch_tensor.min(axis=2)
        norch_result = norch_min.sum()

        norch_result.backward()
        norch_tensor_grad = utils.to_torch(norch_tensor.grad).to(self.device)

        torch_tensor = torch.tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]], requires_grad=True).to(self.device)
        
        torch_min = torch_tensor.min(axis=2)
        torch_result = torch_min.sum()
        torch_result.backward()
        torch_tensor_grad = torch_tensor.grad
        
        self.assertTrue(utils.compare_torch(norch_tensor_grad, torch_tensor_grad))
    

    def test_broadcasted_addition_autograd(self):
        """
        Test autograd for broadcasting addition: tensor1 + tensor2
        """
        norch_tensor1 = norch.Tensor([[[1., 2, 3], [4, 5, 6]]], requires_grad=True).to(self.device)  # Shape (1, 2, 3)
        norch_tensor2 = norch.Tensor([1.5, -1, 0], requires_grad=True).to(self.device)  # Shape (3)
        norch_result = (norch_tensor1 + norch_tensor2).sum()
        norch_result.backward()
        norch_tensor1_grad = utils.to_torch(norch_tensor1.grad).to(self.device)
        norch_tensor2_grad = utils.to_torch(norch_tensor2.grad).to(self.device)

        torch_tensor1 = torch.tensor([[[1., 2, 3], [4, 5, 6]]], requires_grad=True).to(self.device)  # Shape (1, 2, 3)
        torch_tensor2 = torch.tensor([1.5, -1, 0], requires_grad=True).to(self.device)  # Shape (3)
        torch_result = (torch_tensor1 + torch_tensor2).sum()
        torch_result.backward()
        torch_tensor1_grad = torch_tensor1.grad
        torch_tensor2_grad = torch_tensor2.grad

        self.assertTrue(utils.compare_torch(norch_tensor1_grad, torch_tensor1_grad))
        self.assertTrue(utils.compare_torch(norch_tensor2_grad, torch_tensor2_grad))

        ## reversed order broadcasting
        norch_tensor1 = norch.Tensor([[[1., 2, 3], [4, 5, 6]]], requires_grad=True).to(self.device)  # Shape (1, 2, 3)
        norch_tensor2 = norch.Tensor([1.5, -1, 0], requires_grad=True).to(self.device)  # Shape (3)

        norch_result = (norch_tensor2 + norch_tensor1).sum()
        norch_result.backward()
        norch_tensor1_grad = utils.to_torch(norch_tensor1.grad).to(self.device)
        norch_tensor2_grad = utils.to_torch(norch_tensor2.grad).to(self.device)

        torch_tensor1 = torch.tensor([[[1., 2, 3], [4, 5, 6]]], requires_grad=True).to(self.device)  # Shape (1, 2, 3)
        torch_tensor2 = torch.tensor([1.5, -1, 0], requires_grad=True).to(self.device)  # Shape (3)

        torch_result = (torch_tensor2 + torch_tensor1).sum()
        torch_result.backward()
        torch_tensor1_grad = torch_tensor1.grad
        torch_tensor2_grad = torch_tensor2.grad

        self.assertTrue(utils.compare_torch(norch_tensor1_grad, torch_tensor1_grad))
        self.assertTrue(utils.compare_torch(norch_tensor2_grad, torch_tensor2_grad))
    
    def test_subtraction(self):
        """
        Test autograd from subtraction two tensors: tensor1 - tensor2
        """
        norch_tensor1_sub = norch.Tensor([[[1, 2.5], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        norch_tensor2_sub = norch.Tensor([[[1, 1.], [1, 1.9]], [[1, 1], [1, 1]]], requires_grad=True).to(self.device)
        norch_result_sub = (norch_tensor1_sub - norch_tensor2_sub).sum()
        norch_result_sub.backward()
        norch_tensor1_grad_sub = utils.to_torch(norch_tensor1_sub.grad).to(self.device)
        norch_tensor2_grad_sub = utils.to_torch(norch_tensor2_sub.grad).to(self.device)

        torch_tensor1_sub = torch.tensor([[[1, 2.5], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        torch_tensor2_sub = torch.tensor([[[1, 1.], [1, 1.9]], [[1, 1], [1, 1]]], requires_grad=True).to(self.device)
        torch_result_sub = (torch_tensor1_sub - torch_tensor2_sub).sum()
        torch_result_sub.backward()
        torch_tensor1_grad_sub = torch_tensor1_sub.grad
        torch_tensor2_grad_sub = torch_tensor2_sub.grad

        self.assertTrue(utils.compare_torch(norch_tensor1_grad_sub, torch_tensor1_grad_sub))
        self.assertTrue(utils.compare_torch(norch_tensor2_grad_sub, torch_tensor2_grad_sub))
        
    def test_broadcasted_subtraction_autograd(self):
        """
        Test autograd for broadcasting subtraction: tensor1 - tensor2
        """
        norch_tensor1 = norch.Tensor([[[1., 2, 3], [4, 5, 6]]], requires_grad=True).to(self.device)  # Shape (1, 2, 3)
        norch_tensor2 = norch.Tensor([1.5, -1, 0], requires_grad=True).to(self.device)  # Shape (3)
        norch_result = (norch_tensor1 - norch_tensor2).sum()
        norch_result.backward()
        norch_tensor1_grad = utils.to_torch(norch_tensor1.grad).to(self.device)
        norch_tensor2_grad = utils.to_torch(norch_tensor2.grad).to(self.device)

        torch_tensor1 = torch.tensor([[[1., 2, 3], [4, 5, 6]]], requires_grad=True).to(self.device)  # Shape (1, 2, 3)
        torch_tensor2 = torch.tensor([1.5, -1, 0], requires_grad=True).to(self.device)  # Shape (3)
        torch_result = (torch_tensor1 - torch_tensor2).sum()
        torch_result.backward()
        torch_tensor1_grad = torch_tensor1.grad
        torch_tensor2_grad = torch_tensor2.grad

        self.assertTrue(utils.compare_torch(norch_tensor1_grad, torch_tensor1_grad))
        self.assertTrue(utils.compare_torch(norch_tensor2_grad, torch_tensor2_grad))

        # reversed order broadcasting
        norch_tensor1 = norch.Tensor([[[1., 2, 3], [4, 5, 6]]], requires_grad=True).to(self.device)  # Shape (1, 2, 3)
        norch_tensor2 = norch.Tensor([1.5, -1, 0], requires_grad=True).to(self.device)  # Shape (3)

        norch_result = (norch_tensor2 - norch_tensor1).sum()
        norch_result.backward()
        norch_tensor1_grad = utils.to_torch(norch_tensor1.grad).to(self.device)
        norch_tensor2_grad = utils.to_torch(norch_tensor2.grad).to(self.device)

        torch_tensor1 = torch.tensor([[[1., 2, 3], [4, 5, 6]]], requires_grad=True).to(self.device)  # Shape (1, 2, 3)
        torch_tensor2 = torch.tensor([1.5, -1, 0], requires_grad=True).to(self.device)  # Shape (3)
        
        torch_result = (torch_tensor2 - torch_tensor1).sum()
        torch_result.backward()
        torch_tensor1_grad = torch_tensor1.grad
        torch_tensor2_grad = torch_tensor2.grad

        self.assertTrue(utils.compare_torch(norch_tensor1_grad, torch_tensor1_grad))
        self.assertTrue(utils.compare_torch(norch_tensor2_grad, torch_tensor2_grad))
    

    def test_division(self):
        """
        Test autograd from dividing two tensors: tensor1 / tensor2
        """
        norch_tensor1_div = norch.Tensor([[[2, 5.1], [6, -8]], [[10, 12], [14, 16]]], requires_grad=True).to(self.device)
        norch_tensor2_div = norch.Tensor([[[1, 1], [2, 2.2]], [[3, 3], [4, 4]]], requires_grad=True).to(self.device)
        norch_result_div = (norch_tensor1_div / norch_tensor2_div).sum()
        norch_result_div.backward()
        norch_tensor1_grad_div = utils.to_torch(norch_tensor1_div.grad).to(self.device)
        norch_tensor2_grad_div = utils.to_torch(norch_tensor2_div.grad).to(self.device)

        torch_tensor1_div = torch.tensor([[[2, 5.1], [6, -8]], [[10, 12], [14, 16]]], requires_grad=True).to(self.device)
        torch_tensor2_div = torch.tensor([[[1, 1], [2, 2.2]], [[3, 3], [4, 4]]], requires_grad=True).to(self.device)
        torch_result_div = (torch_tensor1_div / torch_tensor2_div).sum()
        torch_result_div.backward()
        torch_tensor1_grad_div = torch_tensor1_div.grad
        torch_tensor2_grad_div = torch_tensor2_div.grad

        self.assertTrue(utils.compare_torch(norch_tensor1_grad_div, torch_tensor1_grad_div))
        self.assertTrue(utils.compare_torch(norch_tensor2_grad_div, torch_tensor2_grad_div))
    
    
    def test_tensor_division_scalar(self):
        """
        Test autograd from dividing tensor by scalar: tensor / scalar
        """
        norch_tensor_div_scalar = norch.Tensor([[[2, 4.7], [6, 8]], [[10, 12], [14, 16]]], requires_grad=True).to(self.device)
        scalar = 2
        norch_result_div_scalar = (norch_tensor_div_scalar / scalar).sum()
        norch_result_div_scalar.backward()
        norch_tensor_grad_div_scalar = utils.to_torch(norch_tensor_div_scalar.grad).to(self.device)

        torch_tensor_div_scalar = torch.tensor([[[2, 4.7], [6, 8]], [[10, 12], [14, 16]]], requires_grad=True).to(self.device)
        torch_result_div_scalar = (torch_tensor_div_scalar / scalar).sum()
        torch_result_div_scalar.backward()
        torch_tensor_grad_div_scalar = torch_tensor_div_scalar.grad

        self.assertTrue(utils.compare_torch(norch_tensor_grad_div_scalar, torch_tensor_grad_div_scalar))
    
    
    def test_scalar_division_tensor(self):
        """
        Test autograd from dividing scalar by tensor: scalar / tensor
        """
        scalar = 2
        norch_tensor_scalar_div = norch.Tensor([[[1, 2.23], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        norch_result_scalar_div = (scalar / norch_tensor_scalar_div).sum()
        norch_result_scalar_div.backward()
        norch_tensor_grad_scalar_div = utils.to_torch(norch_tensor_scalar_div.grad).to(self.device)

        torch_tensor_scalar_div = torch.tensor([[[1, 2.23], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        torch_result_scalar_div = (scalar / torch_tensor_scalar_div).sum()
        torch_result_scalar_div.backward()
        torch_tensor_grad_scalar_div = torch_tensor_scalar_div.grad

        self.assertTrue(utils.compare_torch(norch_tensor_grad_scalar_div, torch_tensor_grad_scalar_div))
    
    
    def test_power_scalar_tensor(self):
        """
        Test autograd from scalar raised to tensor: scalar ** tensor
        """
        scalar = 2
        norch_tensor_power_st = norch.Tensor([[[2, 3.21], [4, 2.1]], [[6, 7], [8, 9]]], requires_grad=True).to(self.device)
        norch_result_power_st = (scalar ** norch_tensor_power_st).sum()
        norch_result_power_st.backward()
        norch_tensor_grad_power_st = utils.to_torch(norch_tensor_power_st.grad).to(self.device)

        torch_tensor_power_st = torch.tensor([[[2, 3.21], [4, 2.1]], [[6, 7], [8, 9]]], requires_grad=True).to(self.device)
        torch_result_power_st = (scalar ** torch_tensor_power_st).sum()
        torch_result_power_st.backward()
        torch_tensor_grad_power_st = torch_tensor_power_st.grad

        self.assertTrue(utils.compare_torch(norch_tensor_grad_power_st, torch_tensor_grad_power_st))
    
    def test_power_tensor_scalar(self):
        """
        Test autograd from tensor raised to scalar: tensor ** scalar
        """
        scalar = 2
        norch_tensor_power_ts = norch.Tensor([[[2, 3], [4, 2.1]], [[6, 7], [8, 9]]], requires_grad=True).to(self.device)
        norch_result_power_ts = (norch_tensor_power_ts ** scalar).sum()
        norch_result_power_ts.backward()
        norch_tensor_grad_power_ts = utils.to_torch(norch_tensor_power_ts.grad).to(self.device)

        torch_tensor_power_ts = torch.tensor([[[2, 3], [4, 2.1]], [[6, 7], [8, 9]]], requires_grad=True).to(self.device)
        torch_result_power_ts = (torch_tensor_power_ts ** scalar).sum()
        torch_result_power_ts.backward()
        torch_tensor_grad_power_ts = torch_tensor_power_ts.grad

        self.assertTrue(utils.compare_torch(norch_tensor_grad_power_ts, torch_tensor_grad_power_ts))

    def test_matmul(self):
        """
        Test autograd from matrix multiplication: matmul(tensor1, tensor2)
        """
        norch_tensor1_matmul = norch.Tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        norch_tensor2_matmul = norch.Tensor([[[1.1, 3], [4, 5]], [[6, 7], [8, 9]]], requires_grad=True).to(self.device)
        norch_result_matmul = (norch_tensor1_matmul @ norch_tensor2_matmul).sum()
        norch_result_matmul.backward()
        norch_tensor1_grad_matmul = utils.to_torch(norch_tensor1_matmul.grad).to(self.device)
        norch_tensor2_grad_matmul = utils.to_torch(norch_tensor2_matmul.grad).to(self.device)

        torch_tensor1_matmul = torch.tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        torch_tensor2_matmul = torch.tensor([[[1.1, 3], [4, 5]], [[6, 7], [8, 9]]], requires_grad=True).to(self.device)
        torch_result_matmul = (torch_tensor1_matmul @ torch_tensor2_matmul).sum()
        torch_result_matmul.backward()
        torch_tensor1_grad_matmul = torch_tensor1_matmul.grad
        torch_tensor2_grad_matmul = torch_tensor2_matmul.grad

        self.assertTrue(utils.compare_torch(norch_tensor1_grad_matmul, torch_tensor1_grad_matmul))
        self.assertTrue(utils.compare_torch(norch_tensor2_grad_matmul, torch_tensor2_grad_matmul))
    
    def test_batched_matmul(self):
        """
        Test autograd from batched matrix multiplication: BxMxP = BxNxM @ BxMxP
        """
        B = 3  # Batch size

        norch_tensor1_matmul = norch.Tensor([[[1., 2], [3, -4], [5, 6], [7, 8]] for _ in range(B)], requires_grad=True).to(self.device)
        norch_tensor2_matmul = norch.Tensor([[[2., 3, 1, 0, 4], [5, -1, 2, 3, 0]] for _ in range(B)], requires_grad=True).to(self.device)

        norch_result_matmul = norch_tensor1_matmul @ norch_tensor2_matmul
        norch_result_matmul_sum = norch_result_matmul.sum()  # Sum over all elements
        norch_result_matmul_sum.backward()

        # Convert gradients to torch tensors
        norch_tensor1_grad_matmul = utils.to_torch(norch_tensor1_matmul.grad).to(self.device)
        norch_tensor2_grad_matmul = utils.to_torch(norch_tensor2_matmul.grad).to(self.device)

        # Repeat the same process with torch tensors
        torch_tensor1_matmul = torch.tensor([[[1., 2], [3, -4], [5, 6], [7, 8]] for _ in range(B)], requires_grad=True).to(self.device)
        torch_tensor2_matmul = torch.tensor([[[2., 3, 1, 0, 4], [5, -1, 2, 3, 0]] for _ in range(B)], requires_grad=True).to(self.device)
        torch_result_matmul = torch.matmul(torch_tensor1_matmul, torch_tensor2_matmul)
        torch_result_matmul_sum = torch_result_matmul.sum()
        torch_result_matmul_sum.backward()
        
        # Extract gradients from torch tensors
        torch_tensor1_grad_matmul = torch_tensor1_matmul.grad
        torch_tensor2_grad_matmul = torch_tensor2_matmul.grad

        # Assertions to compare the gradients
        self.assertTrue(utils.compare_torch(norch_tensor1_grad_matmul, torch_tensor1_grad_matmul))
        self.assertTrue(utils.compare_torch(norch_tensor2_grad_matmul, torch_tensor2_grad_matmul))

    def test_broadcasted_batched_matmul(self):
        """
        Test autograd from broadcasted batched matrix multiplication: BxMxP = NxM @ BxMxP
        """
        B = 3  # Batch size

        norch_tensor1_matmul = norch.Tensor([[1., 2], [3, -4], [5, 6], [7, 8]], requires_grad=True).to(self.device)
        norch_tensor2_matmul = norch.Tensor([[[2., 3, 1, 0, 4], [5, -1, 2, 3, 0]] for _ in range(B)], requires_grad=True).to(self.device)

        norch_result_matmul = norch_tensor1_matmul @ norch_tensor2_matmul
        norch_result_matmul_sum = norch_result_matmul.sum()  # Sum over all elements
        norch_result_matmul_sum.backward()

        # Convert gradients to torch tensors
        norch_tensor1_grad_matmul = utils.to_torch(norch_tensor1_matmul.grad).to(self.device)
        norch_tensor2_grad_matmul = utils.to_torch(norch_tensor2_matmul.grad).to(self.device)

        # Repeat the same process with torch tensors
        torch_tensor1_matmul = torch.tensor([[1., 2], [3, -4], [5, 6], [7, 8]], requires_grad=True).to(self.device)
        torch_tensor2_matmul = torch.tensor([[[2., 3, 1, 0, 4], [5, -1, 2, 3, 0]] for _ in range(B)], requires_grad=True).to(self.device)
        torch_result_matmul = torch.matmul(torch_tensor1_matmul, torch_tensor2_matmul)
        torch_result_matmul_sum = torch_result_matmul.sum()
        torch_result_matmul_sum.backward()
        
        # Extract gradients from torch tensors
        torch_tensor1_grad_matmul = torch_tensor1_matmul.grad
        torch_tensor2_grad_matmul = torch_tensor2_matmul.grad

        # Assertions to compare the gradients
        self.assertTrue(utils.compare_torch(norch_tensor1_grad_matmul, torch_tensor1_grad_matmul))
        self.assertTrue(utils.compare_torch(norch_tensor2_grad_matmul, torch_tensor2_grad_matmul))


    def test_elementwise_mul_scalar(self):
        """
        Test autograd from elementwise multiplication with scalar: scalar * tensor
        """
        scalar = 2
        norch_tensor_elemwise_mul_scalar = norch.Tensor([[[1.1, 2], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        norch_result_elemwise_mul_scalar = (scalar * norch_tensor_elemwise_mul_scalar).sum()
        norch_result_elemwise_mul_scalar.backward()
        norch_tensor_grad_elemwise_mul_scalar = utils.to_torch(norch_tensor_elemwise_mul_scalar.grad).to(self.device)

        torch_tensor_elemwise_mul_scalar = torch.tensor([[[1.1, 2], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        torch_result_elemwise_mul_scalar = (scalar * torch_tensor_elemwise_mul_scalar).sum()
        torch_result_elemwise_mul_scalar.backward()
        torch_tensor_grad_elemwise_mul_scalar = torch_tensor_elemwise_mul_scalar.grad

        self.assertTrue(utils.compare_torch(norch_tensor_grad_elemwise_mul_scalar, torch_tensor_grad_elemwise_mul_scalar))
    
    
    def test_elementwise_mul_tensor(self):
        """
        Test autograd from elementwise multiplication between two tensors: tensor1 * tensor2
        """
        norch_tensor1_elemwise_mul = norch.Tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        norch_tensor2_elemwise_mul = norch.Tensor([[[1.1, 3], [4, 5]], [[6, 7], [8, 9]]], requires_grad=True).to(self.device)
        norch_result_elemwise_mul = (norch_tensor1_elemwise_mul * norch_tensor2_elemwise_mul).sum()
        norch_result_elemwise_mul.backward()
        norch_tensor1_grad_elemwise_mul = utils.to_torch(norch_tensor1_elemwise_mul.grad).to(self.device)
        norch_tensor2_grad_elemwise_mul = utils.to_torch(norch_tensor2_elemwise_mul.grad).to(self.device)

        torch_tensor1_elemwise_mul = torch.tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        torch_tensor2_elemwise_mul = torch.tensor([[[1.1, 3], [4, 5]], [[6, 7], [8, 9]]], requires_grad=True).to(self.device)
        torch_result_elemwise_mul = (torch_tensor1_elemwise_mul * torch_tensor2_elemwise_mul).sum()
        torch_result_elemwise_mul.backward()
        torch_tensor1_grad_elemwise_mul = torch_tensor1_elemwise_mul.grad
        torch_tensor2_grad_elemwise_mul = torch_tensor2_elemwise_mul.grad

        self.assertTrue(utils.compare_torch(norch_tensor1_grad_elemwise_mul, torch_tensor1_grad_elemwise_mul))
        self.assertTrue(utils.compare_torch(norch_tensor2_grad_elemwise_mul, torch_tensor2_grad_elemwise_mul))

    def test_sin_tensor(self):
        """
        Test autograd from sin operation: sin(tensor)
        """
        norch_sin_tensor = norch.Tensor([[[2, 3.21], [4, 2.1]], [[6, 7], [8, 9]]], requires_grad=True).to(self.device)
        norch_result_sin_tensor = (norch_sin_tensor.sin()).sum()
        norch_result_sin_tensor.backward()
        torch_result_sin_tensor_grad = utils.to_torch(norch_sin_tensor.grad).to(self.device)

        torch_sin_tensor = torch.tensor([[[2, 3.21], [4, 2.1]], [[6, 7], [8, 9]]], requires_grad=True).to(self.device)
        torch_expected_sin_tensor = (torch.sin(torch_sin_tensor)).sum()
        torch_expected_sin_tensor.backward()
        torch_expected_sin_tensor_grad = torch_sin_tensor.grad

        self.assertTrue(utils.compare_torch(torch_result_sin_tensor_grad, torch_expected_sin_tensor_grad))
    
    def test_cos_tensor(self):
        """
        Test autograd from cosine operation: cos(tensor)
        """
        norch_cos_tensor = norch.Tensor([[[2, 3.21], [4, 2.1]], [[6, 7], [8, 9]]], requires_grad=True).to(self.device)
        norch_result_cos_tensor = (norch_cos_tensor.sin()).sum()
        norch_result_cos_tensor.backward()
        torch_result_cos_tensor_grad = utils.to_torch(norch_cos_tensor.grad).to(self.device)

        torch_cos_tensor = torch.tensor([[[2, 3.21], [4, 2.1]], [[6, 7], [8, 9]]], requires_grad=True).to(self.device)
        torch_expected_cos_tensor = (torch.sin(torch_cos_tensor)).sum()
        torch_expected_cos_tensor.backward()
        torch_expected_cos_tensor_grad = torch_cos_tensor.grad

        self.assertTrue(utils.compare_torch(torch_result_cos_tensor_grad, torch_expected_cos_tensor_grad))
    
    def test_reshape(self):
        """
        Test autograd from reshaping a tensor: tensor.reshape(shape)
        """
        norch_tensor_reshape = norch.Tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        new_shape = [2, 4]
        norch_result_reshape = norch_tensor_reshape.reshape(new_shape).sum()
        norch_result_reshape.backward()
        norch_tensor_grad_reshape = utils.to_torch(norch_tensor_reshape.grad).to(self.device)

        torch_tensor_reshape = torch.tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        torch_result_reshape = torch_tensor_reshape.reshape(new_shape).sum()
        torch_result_reshape.backward()
        torch_tensor_grad_reshape = torch_tensor_reshape.grad

        self.assertTrue(utils.compare_torch(norch_tensor_grad_reshape, torch_tensor_grad_reshape))
    
    
    def test_transpose_axes(self):
        """
        Test autograd from transposing a tensor with specific axes: tensor.transpose(axis1, axis2)
        """
        norch_tensor_transpose = norch.Tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        axis1, axis2 = 0, 2
        norch_result_transpose = norch_tensor_transpose.transpose(axis1, axis2).sum()
        norch_result_transpose.backward()
        norch_tensor_grad_transpose = utils.to_torch(norch_tensor_transpose.grad).to(self.device)

        torch_tensor_transpose = torch.tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        torch_result_transpose = torch_tensor_transpose.transpose(axis1, axis2).sum()
        torch_result_transpose.backward()
        torch_tensor_grad_transpose = torch_tensor_transpose.grad

        self.assertTrue(utils.compare_torch(norch_tensor_grad_transpose, torch_tensor_grad_transpose))
    
    
    def test_T(self):
        """
        Test autograd from transposing a tensor using .T attribute
        """
        norch_tensor_T = norch.Tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        norch_result_T = norch_tensor_T.T.sum()
        norch_result_T.backward()
        norch_tensor_grad_T = utils.to_torch(norch_tensor_T.grad).to(self.device)

        torch_tensor_T = torch.tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]], requires_grad=True).to(self.device)
        torch_result_T = torch_tensor_T.mT.sum()
        torch_result_T.backward()
        torch_tensor_grad_T = torch_tensor_T.grad

        self.assertTrue(utils.compare_torch(norch_tensor_grad_T, torch_tensor_grad_T))

    def test_reshape_then_matmul(self):
        """
        Test autograd from reshaping a tensor then performing matrix multiplication: matmul(tensor1.reshape(shape), tensor2)
        """
        norch_tensor1 = norch.Tensor([[1, 2.1], [3, -4], [5, 6], [7, 8]], requires_grad=True).to(self.device)
        norch_tensor2 = norch.Tensor([[1, 5.1], [0.1, -4], [0, 6], [7, 8]], requires_grad=True).to(self.device)

        new_shape = [2, 4]

        norch_result_reshape_matmul = (norch_tensor1.reshape(new_shape) @ norch_tensor2).sum()
        norch_result_reshape_matmul.backward()
        norch_tensor_grad_reshape_matmul1 = utils.to_torch(norch_tensor1.grad).to(self.device)
        norch_tensor_grad_reshape_matmul2 = utils.to_torch(norch_tensor2.grad).to(self.device)

        torch_tensor1 = torch.tensor([[1, 2.1], [3, -4], [5, 6], [7, 8]], dtype=torch.float32, requires_grad=True).to(self.device)
        torch_tensor2 = torch.tensor([[1, 5.1], [0.1, -4], [0, 6], [7, 8]], dtype=torch.float32, requires_grad=True).to(self.device)     
        
        torch_result_reshape_matmul = (torch_tensor1.reshape(new_shape) @ torch_tensor2).sum()
        torch_result_reshape_matmul.backward()
        torch_tensor_grad_reshape_matmul1 = torch_tensor1.grad
        torch_tensor_grad_reshape_matmul2 = torch_tensor2.grad
        
        self.assertTrue(utils.compare_torch(norch_tensor_grad_reshape_matmul1, torch_tensor_grad_reshape_matmul1))
        self.assertTrue(utils.compare_torch(norch_tensor_grad_reshape_matmul2, torch_tensor_grad_reshape_matmul2))


    def test_T_then_matmul(self):
        """
        Test autograd from transposing a tensor then performing matrix multiplication: matmul(tensor.T, tensor)
        """
        norch_tensor1 = norch.Tensor([[1, 2.1], [3, -4], [5, 6], [7, 8]], requires_grad=True)
        norch_tensor2 = norch.Tensor([[1, 5.1], [0.1, -4], [0, 6], [7, 8]], requires_grad=True)

        norch_result_T_matmul = (norch_tensor1.T @ norch_tensor2).sum()
        norch_result_T_matmul.backward()
        norch_tensor_grad_T_matmul1 = utils.to_torch(norch_tensor1.grad).to(self.device)
        norch_tensor_grad_T_matmult2 = utils.to_torch(norch_tensor2.grad).to(self.device)

        torch_tensor1 = torch.tensor([[1, 2.1], [3, -4], [5, 6], [7, 8]], dtype=torch.float32, requires_grad=True).to(self.device)
        torch_tensor2 = torch.tensor([[1, 5.1], [0.1, -4], [0, 6], [7, 8]], dtype=torch.float32, requires_grad=True).to(self.device)     
        
        torch_result_T_matmul = (torch_tensor1.T @ torch_tensor2).sum()
        torch_result_T_matmul.backward()
        torch_tensor_grad_T_matmul1 = torch_tensor1.grad
        torch_tensor_grad_T_matmul2 = torch_tensor2.grad
        
        self.assertTrue(utils.compare_torch(norch_tensor_grad_T_matmul1, torch_tensor_grad_T_matmul1))
        self.assertTrue(utils.compare_torch(norch_tensor_grad_T_matmult2, torch_tensor_grad_T_matmul2))

    def todo(self):
        """
        The code has a problem on the following operation
        tensor1.reshape(..) @ tensor1
        print(tensor1.grad)
        (also transpsoe and .T)
        """
        pass

    def test_transpose_axes_then_matmul(self):
        """
        Test autograd from transposing a tensor with specific axes then performing matrix multiplication: matmul(tensor.transpose(axis1, axis2), tensor)
        """
        norch_tensor1 = norch.Tensor([[1, 2.1], [3, -4], [5, 6], [7, 8]], requires_grad=True).to(self.device)
        norch_tensor2 = norch.Tensor([[1, 5.1], [0.1, -4], [0, 6], [7, 8]], requires_grad=True).to(self.device)

        norch_result_transpose_matmul = (norch_tensor1.transpose(0, 1) @ norch_tensor2).sum()
        norch_result_transpose_matmul.backward()
        norch_tensor_grad_transpose_matmul1 = utils.to_torch(norch_tensor1.grad).to(self.device)
        norch_tensor_grad_transpose_matmult2 = utils.to_torch(norch_tensor2.grad).to(self.device)

        torch_tensor1 = torch.tensor([[1, 2.1], [3, -4], [5, 6], [7, 8]], dtype=torch.float32, requires_grad=True).to(self.device)
        torch_tensor2 = torch.tensor([[1, 5.1], [0.1, -4], [0, 6], [7, 8]], dtype=torch.float32, requires_grad=True).to(self.device)  
        
        torch_result_transpose_matmul = (torch_tensor1.T @ torch_tensor2).sum()
        torch_result_transpose_matmul.backward()
        torch_tensor_grad_transpose_matmul1 = torch_tensor1.grad
        torch_tensor_grad_transpose_matmul2 = torch_tensor2.grad
        
        self.assertTrue(utils.compare_torch(norch_tensor_grad_transpose_matmul1, torch_tensor_grad_transpose_matmul1))
        self.assertTrue(utils.compare_torch(norch_tensor_grad_transpose_matmult2, torch_tensor_grad_transpose_matmul2))

if __name__ == '__main__':
    unittest.main()