import os
import norch
import norch.distributed as dist
import norch.distributed

import norch.nn as nn
import norch.optim as optim
from norch.utils.data.dataloader import DataLoader
from norch.nn.parallel import DistributedDataParallel
from norch.utils.data.distributed import DistributedSampler
from norch.norchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(1)

def main():

    BATCH_SIZE = 32
    device = "cpu"
    epochs = 10

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Reshape([-1, 784, 1])
        ]
    )

    target_transform = T.Compose(
        [
            T.ToTensor()
        ]
    )

    train_data, test_data = norch.norchvision.datasets.MNIST.splits(transform=transform, target_transform=target_transform)
    train_loader = norch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.fc1 = nn.Linear(784, 30)
            self.sigmoid1 = nn.Sigmoid()
            self.fc2 = nn.Linear(30, 10)
            self.sigmoid2 = nn.Sigmoid()

        def forward(self, x):
            out = self.fc1(x)
            out = self.sigmoid1(out)
            out = self.fc2(out)
            out = self.sigmoid2(out)
            
            return out

    model = MyModel().to(device)
    model = DistributedDataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_list = []

    for epoch in range(epochs):    
        for idx, batch in enumerate(train_loader):

            inputs, target = batch

            inputs = inputs.to(device)
            target = target.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, target)
            
            optimizer.zero_grad()
            
            print("#####################\n\nantes backward")
            print(model.module.fc1.bias.grad)
            loss.backward()
            print(model.module.fc1.bias.grad)
            print("\n\n\n###############\n\npós backward")


            optimizer.step()
            print("@@")
            break

        break

if __name__ == "__main__":
    main()

