"""import os
import norch
import norch.distributed as dist

def main():

    local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', -1))
    rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))
    world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', -1))

    dist.init_process_group(rank, world_size)

    tensor = norch.Tensor([1,1,1])
    tensor = (rank + 1) * tensor
    print(f"BEFORE on rank {rank}: {tensor} \n\n")

    dist.allreduce_sum_tensor(tensor)

    print(f"AFTER ALLREDUCE on rank {rank}: {tensor} \n\n")

    print("###############\n\n\n")

    tensor = tensor * 10
    print(f"BEFORE BROADCAST on rank {rank}: {tensor} \n\n")

    dist.broadcast_tensor(tensor)

    print(f"AFTER BROADCAST on rank {rank}: {tensor} \n\n")

if __name__ == "__main__":
    main()

"""
import norch
import matplotlib.pyplot as plt
import numpy as np
import random
from norch.norchvision import transforms

train_data, test_data = norch.norchvision.datasets.MNIST.splits(transform=transforms.ToTensor())
train_sampler = norch.utils.data.distributed.DistributedSampler(dataset=train_data, num_replicas=10, rank=2)
train_loader = norch.utils.data.Dataloader(train_data, batch_size = 50, sampler=train_sampler)
input_sample, target_sample = train_data[0]

fig = plt.figure(figsize = (20, 10))
columns = 4
rows = 2

# Choose a random image
for image_index, batch in enumerate(train_loader): 

    image, label = batch
    fig.add_subplot(rows, columns, image_index+1)
    plt.imshow(np.array(image).reshape(28, 28))
    plt.title(label)
    plt.axis('off')
    if image_index > 6:
        break
plt.show()