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

def main():
    
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

if __name__ == "__main__":
    main()

