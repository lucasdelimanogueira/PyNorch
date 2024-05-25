import numpy as np
from .batch import Batch


class Dataloader:
    
    def __init__(self, dataset, batch_size=32, sampler=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler

    def __iter__(self):
        if self.sampler is not None:
            indices = list(self.sampler)
        else:
            indices = np.arange(len(self.dataset))
        
        for start in range(0, len(indices), self.batch_size):
            end = start + self.batch_size
            batch_indices = indices[start:end]
            batch_size = len(batch_indices)
            yield Batch([self.dataset[i] for i in batch_indices], batch_size)

    def __len__(self):
        if self.sampler is not None:
            return len(self.sampler) // self.batch_size
        return len(self.dataset) // self.batch_size
