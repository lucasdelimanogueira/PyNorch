# PyNorch
Recreating PyTorch from scratch (C/C++, CUDA and Python, with multi-GPU support and automatic differentiation!)

Project details explanations can also be found on [medium](https://towardsdatascience.com/recreating-pytorch-from-scratch-with-gpu-support-and-automatic-differentiation-8f565122a3cc).

# 1 - About
**PyNorch** is a deep learning framework constructed using C/C++, CUDA and Python. This is a personal project with educational purpose only! `Norch` means **NOT** PyTorch, and we have **NO** claims to rivaling the already established PyTorch. The main objective of **PyNorch** was to give a brief understanding of how a deep learning framework works internally. It implements the Tensor object, multi-GPU support and an automatic differentiation system. 

# 2 - Installation
Install this package from PyPi (you can test on Colab! Also tested on AWS g4dn.12xlarge instance with image ami-061debf863768593d)

```css
$ pip install norch
```

or from cloning this repository
```css
$ git clone https://github.com/lucasdelimanogueira/PyNorch.git
$ cd PyNorch
$ pip install . -v
```

# 3 - Get started
### 3.1 - Tensor operations
```python
import norch

x1 = norch.Tensor([[1, 2], 
                  [3, 4]], requires_grad=True).to("cuda")

x2 = norch.Tensor([[4, 3], 
                  [2, 1]], requires_grad=True).to("cuda)

x3 = x1 @ x2
result = x3.sum()
result.backward

print(x1.grad)
```

### 3.2 - Create a model

```python
import norch
import norch.nn as nn
import norch.optim as optim

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        
        return out
```

### 3.3 - Example single GPU training
```python
# examples/train_singlegpu.py

import norch
from norch.utils.data.dataloader import DataLoader
from norch.norchvision import transforms as T
import norch
import norch.nn as nn
import norch.optim as optim
import random
random.seed(1)

BATCH_SIZE = 32
device = "cuda" #cpu
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
train_loader = DataLoader(train_data, batch_size = BATCH_SIZE)

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
        
        loss.backward()

        optimizer.step()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss[0]:.4f}')
    loss_list.append(loss[0])

```

### 3.4 - Example multi-GPU training
First create a file .py as the example below

```python
# examples/train_multigpu.py

import os
import norch
import norch.distributed as dist
import norch.distributed
import norch.nn as nn
import norch.optim as optim
from norch.nn.parallel import DistributedDataParallel
from norch.utils.data.distributed import DistributedSampler
from norch.norchvision import transforms as T
import random
random.seed(1)

local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', -1))
rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))
world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', -1))

dist.init_process_group(
    rank, 
    world_size
)

BATCH_SIZE = 32
device = local_rank
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
distributed_sampler = DistributedSampler(dataset=train_data, num_replicas=world_size, rank=rank)
train_loader = norch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=distributed_sampler)

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

print(f"Starting training on Rank {rank}/{world_size}\n\n")

for epoch in range(epochs):    
    for idx, batch in enumerate(train_loader):

        inputs, target = batch

        inputs = inputs.to(device)
        target = target.to(device)

        outputs = model(inputs)
        
        loss = criterion(outputs, target)
        
        optimizer.zero_grad()
        
        loss.backward()

        optimizer.step()
    
    if rank == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss[0]:.4f}')
        loss_list.append(loss[0])
```
Then you can run using

```css
$ python3 -m norch.distributed.run --nproc_per_node 4 examples/train_multigpu.py
```



# 4 - Progress

| Development                  | Status      | Feature                                                                |
| ---------------------------- | ----------- | ---------------------------------------------------------------------- |
| Operations                   | in progress | <ul><li>[X] GPU Support</li><li>[X] Autograd</li><li>[X] Broadcasting</li><li>[ ] Memory Management</li></ul>                 |
| Loss                         | in progress | <ul><li>[x] MSE</li><li>[X] Cross Entropy</li></ul>    |
| Data                         | in progress    | <ul><li>[X] Dataset</li><li>[X] Batch</li><li>[X] Iterator</li></ul>   |
| Convolutional Neural Network | in progress    | <ul><li>[ ] Conv2d</li><li>[ ] MaxPool2d</li><li>[ ] Dropout</li></ul> |
| Distributed                  | in progress | <ul><li>[X] All-reduce</li><li>[X] Broadcast</li><li>[X] DistributedSampler</li><li>[X] DistributedDataParallel</li></ul>                 |
