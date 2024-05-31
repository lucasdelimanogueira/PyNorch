import os
import norch
import norch.distributed as dist
import norch.distributed

def main():

    local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', -1))
    rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))
    world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', -1))

    dist.init_process_group(rank, world_size)

    tensor = norch.Tensor([1,1,1]).to(rank)
    tensor = (rank + 1) * tensor
    print(f"BEFORE on rank {rank}: {tensor} \n\n")

    dist.allreduce_sum_tensor(tensor)

    print(f"AFTER ALLREDUCE on rank {rank}: {tensor} \n\n")

    print("###############\n\n\n")

    tensor = tensor * 10
    print(f"BEFORE BROADCAST on rank {rank}: {tensor} \n\n")

    dist.broadcast_tensor(tensor)

    print(f"AFTER BROADCAST on rank {rank}: {tensor} \n\n")

def main2():
    import norch
    import norch.nn as nn
    import norch.optim as optim
    from norch.utils.data.dataloader import DataLoader
    from norch.nn.parallel import DistributedDataParallel
    from norch.norchvision import transforms as T
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    random.seed(1)

    local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', -1))
    rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))
    world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', -1))

    dist.init_process_group(rank, world_size)

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
    distributed_sampler = norch.utils.data.distributed.DistributedSampler(dataset=train_data, num_replicas=world_size, rank=local_rank)
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

    print(f"Local rank: {local_rank}")
    print(f"World size: {world_size}")

    for epoch in range(epochs):    
        for idx, batch in enumerate(train_loader):

            inputs, target = batch

            inputs = inputs.to(device)
            target = target.to(device)

            outputs = model(inputs)
            
            loss = criterion(outputs, target)
            
            optimizer.zero_grad()
            
            loss.backward()
            print(f"AFTER rank {local_rank}: {model.module.fc1.weight.grad}")


            optimizer.step()
            break

        break

if __name__ == "__main__":
    main()

