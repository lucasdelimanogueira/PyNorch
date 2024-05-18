import numpy as np
from .batch import Batch


class Dataloader(object):

    def __init__(self, dataset, batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        starts = np.arange(0, len(self.dataset), self.batch_size)

        for start in starts:
            end = start + self.batch_size
            batch_size = min(end, len(self.dataset)) - start
            yield Batch(self.dataset[start:end], batch_size)

    def __len__(self):
        return len(self.dataset) // self.batch_size
