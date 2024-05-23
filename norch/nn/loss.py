from .module import Module
from norch.autograd.functions import *
import norch
from abc import ABC

class Loss(Module, ABC):
    "Abstract class for loss functions"

    def __init__(self):
        super().__init__()

    def forward(self, predictions, labels):
        raise NotImplementedError
    
    def __call__(self, *inputs):
        return self.forward(*inputs)
    

class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, target):
        assert target.shape == predictions.shape, \
            "Labels and predictions shape does not match: {} and {}".format(target.shape, predictions.shape)
        
        cost = ((predictions - target) ** 2).sum() / predictions.numel
        return cost
    

class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        assert isinstance(input, norch.Tensor), \
            "Cross entropy argument 'input' must be Tensor, not {}".format(type(input))
        
        assert isinstance(target, norch.Tensor), \
            "Cross entropy argument 'target' must be Tensor, not {}".format(type(target))
        if input.ndim > 2:
            input = input.squeeze(-1)

        if input.ndim == 1:
            if target.numel == 1:
                num_classes = input.shape[0]
                target = norch.one_hot_encode(target, num_classes).to(target.device)
                
                logits = norch.softmax(input, dim=0)
                target = target.reshape(logits.shape)
                cost = -(logits.log() * target).sum()
                
            else:
                # target -> class probabilities (one-hot encoded)
                assert target.shape == input.shape, \
                    "Input and target shape does not match: {} and {}".format(input.shape, target.shape)
                logits = norch.softmax(input, dim=0)
                target = target.reshape(logits.shape)
                cost = -(logits.log() * target).sum()


        elif input.ndim == 2:
            if target.ndim > 1:
                target = target.squeeze(-1)
            # batched 
            if target.ndim == 1:
                # target -> Ground truth class indices:
                num_classes = input.shape[1]

                target = norch.one_hot_encode(target, num_classes)
                
                batch_size = input.shape[0]
                logits = norch.softmax(input, dim=1)
                target = target.reshape(logits.shape)
                cost = -(logits.log() * target).sum() / batch_size

            else:
                # target -> class probabilities (one-hot encoded)
                assert target.shape == input.shape, \
                    "Input and target shape does not match: {} and {}".format(input.shape, target.shape)
                
                batch_size = input.shape[0]
                logits = norch.softmax(input, dim=1)
                target = target.reshape(logits.shape)
                cost = -(logits.log() * target).sum() / batch_size

        if input.requires_grad:
            cost.grad_fn = CrossEntropyLossBackward(input, target)

        return cost
            


