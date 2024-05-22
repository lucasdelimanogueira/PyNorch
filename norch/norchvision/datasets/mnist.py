import gzip
import os
import norch
from norch.utils.data import Dataset
import numpy as np

class MNIST(Dataset):
    """
    Loads training, validation, and test partitions of the mnist dataset
    (http://yann.lecun.com/exdb/mnist/). If the data is not already contained in data_dir, it will
    try to download it.

    This dataset contains 60000 training examples, and 10000 test examples of handwritten digits
    in {0, ..., 9} and corresponding labels. Each handwritten image has an "original" dimension of
    28x28x1, and is stored row-wise as a string of 784x1 bytes. Pixel values are in range 0 to 255
    (inclusive).

    Args:
        data_dir: String. Relative or absolute path of the dataset.
        devel_size: Integer. Size of the development (validation) dataset partition.

    Returns:
        X_train: float64 numpy array with shape [784, 60000-devel_size] with values in [0, 1].
        Y_train: uint8 numpy array with shape [60000-devel_size]. Labels.
        X_devel: float64 numpy array with shape [784, devel_size] with values in [0, 1].
        Y_devel: uint8 numpy array with shape [devel_size]. Labels.
        X_test: float64 numpy array with shape [784, 10000] with values in [0, 1].
        Y_test: uint8 numpy array with shape [10000]. Labels.
    """

    urls = ['https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
            'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
            'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
            'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',]
    name = 'mnist-data-py'
    dirname = 'mnist'

    def __init__(self, path_data, path_label, transform=None, target_transform=None):
        self.data = self._load_mnist(path_data, header_size=16).reshape((-1, 28, 28))
        self.labels = self._load_mnist(path_label, header_size=8)
        self.transform = transform
        self.target_transform = target_transform

    def _load_mnist(self, path, header_size):
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=header_size)
        return np.asarray(data, dtype=np.uint8)


    @classmethod
    def splits(cls, root='.data', train_data='train-images-idx3-ubyte.gz', train_label='train-labels-idx1-ubyte.gz',
               test_data='t10k-images-idx3-ubyte.gz', test_label='t10k-labels-idx1-ubyte.gz', **kwargs):
        r"""
        Loads training and test partitions of the [mnist dataset](https://www.cs.toronto.edu/~kriz/cifar.html). If
        the data is not already contained in the ``root`` folder, it will download it.

        Args:
            root (str): relative or absolute path of the dataset.

        Returns:
            tuple(Dataset): training and testing datasets
        """
        path = os.path.join(root, cls.dirname, cls.name)
        if not os.path.isdir(path):
            path = cls.download(root)
        train_data = os.path.join(path, train_data)
        train_label = os.path.join(path, train_label)
        test_data = os.path.join(path, test_data)
        test_label = os.path.join(path, test_label)
        return MNIST(train_data, train_label, **kwargs), MNIST(test_data, test_label, **kwargs)

    def __getitem__(self, item):
        data = self.data[item].tolist()
        label = self.labels[item].tolist()

        if self.transform is not None:
            data = self.transform(data)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        

        return data, label

    def __setitem__(self, key, value):
        self.data[key], self.labels[key] = value

    def __len__(self):
        return len(self.data)
