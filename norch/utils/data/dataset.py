from abc import ABC, abstractmethod
import os
from norch.utils import extract_to_dir, download_from_url
from .example import Example
import norch


class Dataset(ABC):
    r"""
    Abstract Dataset class. All dataset for machine learning purposes can inherits from this architecture,
    for convenience.
    """
    urls = []
    name = ''
    dirname = ''

    def __init__(self, examples, fields):
        self.examples = examples
        self.fields = fields

    @classmethod
    def splits(cls, train=None, test=None, valid=None, root='.'):
        raise NotImplementedError

    @classmethod
    def download(cls, root):
        r"""Download and unzip a web archive (.zip, .gz, or .tgz).

        Args:
            root (str): Folder to download data to.

        Returns:
            string: Path to extracted dataset.
        """
        path_dirname = os.path.join(root, cls.dirname)
        path_name = os.path.join(path_dirname, cls.name)
        if not os.path.isdir(path_dirname):
            for url in cls.urls:
                filename = os.path.basename(url)
                zpath = os.path.join(path_dirname, filename)
                if not os.path.isfile(zpath):
                    if not os.path.exists(os.path.dirname(zpath)):
                        os.makedirs(os.path.dirname(zpath))
                    print(f'Download {filename} from {url} to {zpath}')
                    download_from_url(url, zpath)
                extract_to_dir(zpath, path_name)

        return path_name

    def __repr__(self):
        name = self.__class__.__name__
        string = f"Dataset {name}("
        tab = "   "
        for (key, value) in self.__dict__.items():
            if key[0] != "_":
                if isinstance(value, Example):
                    fields = self.fields
                    for (name, field) in fields:
                        string += f"\n{tab}({name}): {field.__class__.__name__}" \
                                  f"(transform={True if field.transform is not None else None}, dtype={field.dtype})"
                elif isinstance(value, norch.Tensor):
                    string += f"\n{tab}({key}): {value.__class__.__name__}(shape={value.shape}, dtype={value.dtype})"
                else:
                    string += f"\n{tab}({key}): {value.__class__.__name__}"
        return f'{string}\n)'

    def __getitem__(self, item):
        return self.examples[item]

    def __setitem__(self, key, value):
        self.examples[key] = value

    def __len__(self):
        return len(self.examples)
