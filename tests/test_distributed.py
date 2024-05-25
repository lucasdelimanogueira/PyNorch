import unittest
import norch
from norch.utils import utils_unittests as utils
import torch
import torchvision
import numpy as np
from norch.norchvision import transforms as norch_transforms
from torchvision import transforms as torch_transforms

class TestDistributed(unittest.TestCase):
    def test_distributed_sampler_batch_1(self):
        transforms = norch_transforms.Compose(
            [
                norch_transforms.ToTensor(),
            ]
        )
        train_data, test_data = norch.norchvision.datasets.MNIST.splits(transform=transforms, target_transform=transforms)
        distributed_sampler = norch.utils.data.distributed.DistributedSampler(dataset=train_data, num_replicas=8, rank=2)
        train_loader = norch.utils.data.DataLoader(train_data, batch_size = 1, sampler=distributed_sampler)

        labels_norch = []
        for i, batch in enumerate(train_loader): 

            image, label = batch
            labels_norch.append(utils.to_torch(label))

            if i > 10:
                break

        transforms = torch_transforms.Compose(
            [
                torch_transforms.ToTensor(),
            ]
        )

        train_data = torchvision.datasets.MNIST(root='./.data/', download=True, transform=transforms)
        distributed_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_data, num_replicas=8, rank=2, shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size = 1, sampler=distributed_sampler)

        labels_torch = []
        for i, batch in enumerate(train_loader): 

            image, label = batch
            labels_torch.append(label)

            if i > 10:
                break

        for label_norch, label_torch in zip(labels_norch, labels_torch):
            self.assertTrue(utils.compare_torch(label_norch, label_torch))

    
    def test_distributed_sampler_batch_32(self):
        transforms = norch_transforms.Compose(
            [
                norch_transforms.ToTensor(),
            ]
        )
        train_data, test_data = norch.norchvision.datasets.MNIST.splits(transform=transforms, target_transform=transforms)
        distributed_sampler = norch.utils.data.distributed.DistributedSampler(dataset=train_data, num_replicas=8, rank=2)
        train_loader = norch.utils.data.DataLoader(train_data, batch_size = 32, sampler=distributed_sampler)

        labels_norch = []
        for i, batch in enumerate(train_loader): 

            image, label = batch
            labels_norch.append(utils.to_torch(label))

            if i > 10:
                break

        transforms = torch_transforms.Compose(
            [
                torch_transforms.ToTensor(),
            ]
        )

        train_data = torchvision.datasets.MNIST(root='./.data/', download=True, transform=transforms)
        distributed_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_data, num_replicas=8, rank=2, shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size = 32, sampler=distributed_sampler)

        labels_torch = []
        for i, batch in enumerate(train_loader): 

            image, label = batch
            labels_torch.append(label)

            if i > 10:
                break

        for label_norch, label_torch in zip(labels_norch, labels_torch):
            print(label_norch, label_torch)
            self.assertTrue(utils.compare_torch(label_norch, label_torch))


        

            
