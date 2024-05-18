class Batch(object):
    """
    A ``Batch``depends on a ``Dataset`` and is made of ``batch_size`` ``Example``.
    """
    def __init__(self, example, batch_size):
        self.example = example
        self.batch_size = batch_size
        
    def __getitem__(self, item):
        return self.example[item]

    def __len__(self):
        return self.batch_size