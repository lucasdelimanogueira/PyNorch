# File: example.py
# Creation: Wednesday August 19th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


class Field(object):
    r"""
    A ``Field`` defines the data to process from a raw dataset. It will convert the data into a tensor. The data can
    be a string, integers, float etc. The data is meant to be preprocessed and the attribute ``transform`` handles
    the way the user want to process the raw data.
    """

    def __init__(self, transform=None, dtype=None):
        self.transform = transform
        self.dtype = dtype

    def process(self, value):
        """Applies the transformation and changes the data's type if necessary.

        Args:
            value (Tensor):

        Returns:

        """
        if self.transform is not None:
            value = self.transform(value)
        if self.dtype is not None:
            value = value.astype(self.dtype)
        return value


class Example(object):
    r"""
    Store a single training / testing example, and store it as an attribute.
    Highly inspired from PyTorch `Example <https://github.com/pytorch/text/blob/master/torchtext/data/example.py>`__.
    
    """
    @classmethod
    def fromlist(cls, values, fields):
        """
        Add an example from a list of data with respect to the fields.

        Args:
            values: raw data
            fields (tuple(string, Field)): fields to preprocess the data on

        Returns:
            None
        """
        example = cls()
        for (value, field) in zip(values, fields):
            assert len(field) == 2, f"expected a field template similar to \
                                    ('name_field', Field()) but got {format(field)}"
            assert isinstance(field[1], Field), f"expected a field template similar to \
                                                ('name_field', Field()) but got {format(field)}"
            name = field[0]
            value = field[1].process(value)
            setattr(example, name, value)

        return example
