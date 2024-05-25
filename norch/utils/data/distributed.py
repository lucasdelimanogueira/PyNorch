import math
import numpy as np

class DistributedSampler:
    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # Create indices for the dataset
        indices = np.arange(len(self.dataset))

        # Add extra samples to make it evenly divisible
        indices = np.concatenate([indices, indices[:(self.total_size - len(indices)) % len(indices)]])
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples