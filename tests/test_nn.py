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

        print(f"Running tests on: {self.device}")

    def test_mse_loss(self):
        """
        Test the MSELoss
        """
        loss_fn_norch = norch.nn.MSELoss()
        loss_fn_torch = torch.nn.MSELoss()

        # Test case 1: Predictions and labels are equal
        predictions_norch = norch.Tensor([[1.1, 2, 3, 4], [1.1, 2, 3, 4]]).to(self.device)
        labels_norch = norch.Tensor([[1.1, 2, 3, 4], [1.1, 2, 3, 3]]).to(self.device)
        loss_norch = loss_fn_norch.forward(predictions_norch, labels_norch)
        loss_torch_result = utils.to_torch(loss_norch).to(self.device)

        predictions_torch = torch.tensor([[1.1, 2, 3, 4], [1.1, 2, 3, 4]]).to(self.device)
        labels_torch = torch.tensor([[1.1, 2, 3, 4], [1.1, 2, 3, 3]]).to(self.device)
        loss_torch_expected = loss_fn_torch(predictions_torch, labels_torch)        
        
        self.assertTrue(utils.compare_torch(loss_torch_result, loss_torch_expected))
        
        # Test case 2: Predictions and labels are different
        predictions_norch = norch.Tensor([1.1, 2, 3, 4]).to(self.device)
        labels_norch = norch.Tensor([4, 3, 2.1, 1]).to(self.device)
        loss_norch = loss_fn_norch.forward(predictions_norch, labels_norch)
        loss_torch_result = utils.to_torch(loss_norch).to(self.device)

        predictions_torch = torch.tensor([1.1, 2, 3, 4]).to(self.device)
        labels_torch = torch.tensor([4, 3, 2.1, 1]).to(self.device)
        loss_torch_expected = loss_fn_torch(predictions_torch, labels_torch)        
        
        self.assertTrue(utils.compare_torch(loss_torch_result, loss_torch_expected))

    def test_cross_entropy_loss(self):
        """
        Test the CrossEntropyLoss
        """
        loss_fn_norch = norch.nn.CrossEntropyLoss()
        loss_fn_torch = torch.nn.CrossEntropyLoss()

        # Test case 1: Single class, single sample
        predictions_norch = norch.Tensor([2.0, 1.0, 0.1]).to(self.device)
        labels_norch = norch.Tensor([0]).to(self.device)
        loss_norch = loss_fn_norch.forward(predictions_norch, labels_norch)

        loss_torch_result = utils.to_torch(loss_norch).to(self.device)

        predictions_torch = torch.tensor([2.0, 1.0, 0.1]).to(self.device)
        labels_torch = torch.tensor(0).to(self.device)
        loss_torch_expected = loss_fn_torch(predictions_torch, labels_torch)

        self.assertTrue(utils.compare_torch(loss_torch_result, loss_torch_expected))

        # Test case 2: Multiple classes, multiple samples
        predictions_norch = norch.Tensor([[0.5, 1.5, 2.5], [1.0, 2.0, 3.0]]).to(self.device)
        labels_norch = norch.Tensor([2, 1]).to(self.device)
        loss_norch = loss_fn_norch.forward(predictions_norch, labels_norch)
        loss_torch_result = utils.to_torch(loss_norch).to(self.device)
        

        predictions_torch = torch.tensor([[0.5, 1.5, 2.5], [1.0, 2.0, 3.0]]).to(self.device)
        labels_torch = torch.tensor([2, 1]).to(self.device)
        loss_torch_expected = loss_fn_torch(predictions_torch, labels_torch)
        
        self.assertTrue(utils.compare_torch(loss_torch_result, loss_torch_expected))
        
        # Test case 3: Edge case - all predictions are zero
        predictions_norch = norch.Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).to(self.device)
        labels_norch = norch.Tensor([1, 2]).to(self.device)
        loss_norch = loss_fn_norch.forward(predictions_norch, labels_norch)
        loss_torch_result = utils.to_torch(loss_norch).to(self.device)
        
        predictions_torch = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).to(self.device)
        labels_torch = torch.tensor([1, 2]).to(self.device)
        loss_torch_expected = loss_fn_torch(predictions_torch, labels_torch)
        
        self.assertTrue(utils.compare_torch(loss_torch_result, loss_torch_expected))

        # Test case 4: Class probabilities instead of class index
        predictions_norch = norch.Tensor([0.5, 0.2, 0.1]).to(self.device)
        labels_norch = norch.Tensor([1., 0, 0]).to(self.device)
        loss_norch = loss_fn_norch.forward(predictions_norch, labels_norch)
        loss_torch_result = utils.to_torch(loss_norch).to(self.device)
        
        predictions_torch = torch.tensor([0.5, 0.2, 0.1]).to(self.device)
        labels_torch = torch.tensor([1., 0, 0]).to(self.device)
        loss_torch_expected = loss_fn_torch(predictions_torch, labels_torch)
        
        self.assertTrue(utils.compare_torch(loss_torch_result, loss_torch_expected))

        # Test case 4: Batched class probabilities instead of class index
        predictions_norch = norch.Tensor([[0.5, 0.2, 0.1], [0.1, 0.5, 0.7]]).to(self.device)
        labels_norch = norch.Tensor([[1., 0, 0], [0, 1, 0]]).to(self.device)
        loss_norch = loss_fn_norch.forward(predictions_norch, labels_norch)
        loss_torch_result = utils.to_torch(loss_norch).to(self.device)

        predictions_torch = torch.tensor([[0.5, 0.2, 0.1], [0.1, 0.5, 0.7]]).to(self.device)
        labels_torch = torch.tensor([[1., 0, 0], [0, 1, 0]]).to(self.device)
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
        x = norch.Tensor([[1, 2, 3]]).to(self.device)
        sigmoid_norch = sigmoid_fn_norch.forward(x)
        sigmoid_torch_result = utils.to_torch(sigmoid_norch).to(self.device)

        x = torch.tensor([[1, 2, 3]]).to(self.device)
        sigmoid_torch_expected = sigmoid_fn_torch.forward(x)

        self.assertTrue(utils.compare_torch(sigmoid_torch_result, sigmoid_torch_expected))

        # Test case 1: Negative input
        x = norch.Tensor([-1, 2, -3]).to(self.device)
        sigmoid_norch = sigmoid_fn_norch.forward(x)
        sigmoid_torch_result = utils.to_torch(sigmoid_norch).to(self.device)

        x = torch.tensor([-1, 2, -3]).to(self.device)
        sigmoid_torch_expected = sigmoid_fn_torch.forward(x)

        self.assertTrue(utils.compare_torch(sigmoid_torch_result, sigmoid_torch_expected))

        # Test case 1: Zero input
        x = norch.Tensor([0, 0, 0]).to(self.device)
        sigmoid_norch = sigmoid_fn_norch.forward(x)
        sigmoid_torch_result = utils.to_torch(sigmoid_norch).to(self.device)

        x = torch.tensor([0, 0, 0]).to(self.device)
        sigmoid_torch_expected = sigmoid_fn_torch.forward(x)

        self.assertTrue(utils.compare_torch(sigmoid_torch_result, sigmoid_torch_expected))

    def test_softmax_activation(self):
        """
        Test Softmax activation function
        """    

        # Test different axes
        axes = [0, 1, 2, -1]

        # Define the input tensors for different test cases
        test_cases = [
            (norch.Tensor([[[1., 2, 3], [4, 5, 6]]]), torch.tensor([[[1., 2, 3], [4, 5, 6]]])),
            (norch.Tensor([[[1., -1, 0], [2, -2, 0]]]), torch.tensor([[[1., -1, 0], [2, -2, 0]]])),
            (norch.Tensor([[[0., 0, 0], [0, 0, 0]]]), torch.tensor([[[0., 0, 0], [0, 0, 0]]]))
        ]

        for dim in axes:
            softmax_fn_norch = norch.nn.Softmax(dim=dim)
            softmax_fn_torch = torch.nn.Softmax(dim=dim)

            for norch_input, torch_input in test_cases:
                # Move tensors to the correct device
                norch_input = norch_input.to(self.device)
                torch_input = torch_input.to(self.device)

                # Forward pass using norch
                softmax_norch = softmax_fn_norch.forward(norch_input)
                softmax_torch_result = utils.to_torch(softmax_norch).to(self.device)

                # Forward pass using torch
                softmax_torch_expected = softmax_fn_torch.forward(torch_input)

                # Compare the results
                self.assertTrue(utils.compare_torch(softmax_torch_result, softmax_torch_expected))

