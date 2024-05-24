import os
import ctypes
from norch import Tensor

def init_process_group(rank, world_size, backend='nccl'):

    Tensor._C.init_process_group.argtypes = [ctypes.c_int, ctypes.c_int]
    Tensor._C.init_process_group.restype = None

    Tensor._C.init_process_group(rank, world_size)

