import numpy as np
from .batch import Batch


class DataLoader:
    
    def __init__(self, dataset, batch_size, sampler=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler

    def __iter__(self):
        if self.sampler is not None:
            indices = list(iter(self.sampler))
    
        else:
            indices = list(range(len(self.dataset)))
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = self.dataset[batch_indices]
            yield Batch(batch_data, len(batch_data))
            
    def __len__(self):
        if self.sampler is not None:
            return len(self.sampler) // self.batch_size
        
        return len(self.dataset) // self.batch_size
