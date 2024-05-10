import unittest
import norch
from norch.utils import utils_unittests as utils
import torch
import os

class TestNNModuleLoss(unittest.TestCase):

    def setUp(self):
        self.device = os.environ.get('device')
        if self.device is None or self.device != 'cuda':
            self.device = 'cpu'

    def test_mse_loss(self):
        """
        Test the MSELoss
        """
        loss_fn_norch = norch.nn.MSELoss()
        loss_fn_torch = torch.nn.MSELoss()

        # Test case 1: Predictions and labels are equal
        predictions_norch = norch.Tensor([1.1, 2, 3, 4])
        labels_norch = norch.Tensor([1.1, 2, 3, 4])
        loss_norch = loss_fn_norch.forward(predictions_norch, labels_norch)
        loss_torch_result = utils.to_torch(loss_norch)

        predictions_torch = torch.tensor([1.1, 2, 3, 4])
        labels_torch = torch.tensor([1.1, 2, 3, 4])
        loss_torch_expected = loss_fn_torch(predictions_torch, labels_torch)        
        
        self.assertTrue(utils.compare_torch(loss_torch_result, loss_torch_expected))
        
        # Test case 2: Predictions and labels are different
        predictions_norch = norch.Tensor([1.1, 2, 3, 4])
        labels_norch = norch.Tensor([4, 3, 2.1, 1])
        loss_norch = loss_fn_norch.forward(predictions_norch, labels_norch)
        loss_torch_result = utils.to_torch(loss_norch)

        predictions_torch = torch.tensor([1.1, 2, 3, 4])
        labels_torch = torch.tensor([4, 3, 2.1, 1])
        loss_torch_expected = loss_fn_torch(predictions_torch, labels_torch)        
        
        self.assertTrue(utils.compare_torch(loss_torch_result, loss_torch_expected))


class TestNNModuleActivationFn(unittest.TestCase):

    def setUp(self):
        self.device = os.environ.get('device')
        if self.device is None or self.device != 'cuda':
            self.device = 'cpu'

    def test_sigmoid_activation(self):
        """
        Test Sigmoid activation function
        """
        sigmoid_fn_norch = norch.nn.Sigmoid()
        sigmoid_fn_torch = torch.nn.Sigmoid()

        # Test case 1: Positive input
        x = norch.Tensor([1, 2, 3])
        sigmoid_norch = sigmoid_fn_norch.forward(x)
        sigmoid_torch_result = utils.to_torch(sigmoid_norch)

        x = torch.tensor([1, 2, 3])
        sigmoid_torch_expected = sigmoid_fn_torch.forward(x)

        self.assertTrue(utils.compare_torch(sigmoid_torch_result, sigmoid_torch_expected))

        # Test case 1: Negative input
        x = norch.Tensor([-1, 2, -3])
        sigmoid_norch = sigmoid_fn_norch.forward(x)
        sigmoid_torch_result = utils.to_torch(sigmoid_norch)

        x = torch.tensor([-1, 2, -3])
        sigmoid_torch_expected = sigmoid_fn_torch.forward(x)

        self.assertTrue(utils.compare_torch(sigmoid_torch_result, sigmoid_torch_expected))

        # Test case 1: Zero input
        x = norch.Tensor([0, 0, 0])
        sigmoid_norch = sigmoid_fn_norch.forward(x)
        sigmoid_torch_result = utils.to_torch(sigmoid_norch)

        x = torch.tensor([0, 0, 0])
        sigmoid_torch_expected = sigmoid_fn_torch.forward(x)

        self.assertTrue(utils.compare_torch(sigmoid_torch_result, sigmoid_torch_expected))