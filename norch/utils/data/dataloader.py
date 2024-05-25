import numpy as np
from .batch import Batch


class Dataloader:
    
    def __init__(self, dataset, batch_size=32, sampler=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler

    def __iter__(self):
        if self.sampler is not None:
            indices = iter(self.sampler)

        else:
            indices = range(len(self.dataset))

        for idx in indices:
            start = idx * self.batch_size
            end = min(start + self.batch_size, len(self.dataset))
            yield Batch(self.dataset[start:end], end - start)


    def __len__(self):
        if self.sampler is not None:
            return len(self.sampler) // self.batch_size
        
        return len(self.dataset) // self.batch_size
