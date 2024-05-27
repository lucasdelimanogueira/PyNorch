import os
import ctypes
from norch import Tensor, CTensor

def init_process_group(rank, world_size, backend='nccl'):

    Tensor._C.init_process_group.argtypes = [ctypes.c_int, ctypes.c_int]
    Tensor._C.init_process_group.restype = None

    Tensor._C.init_process_group(rank, world_size)

def get_rank():
    return os.getenv('OMPI_COMM_WORLD_RANK', 0)

def get_world_size():
    return os.getenv('OMPI_COMM_WORLD_SIZE', 1)

def broadcast_tensor(tensor, src=0):

    Tensor._C.broadcast_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.c_int]
    Tensor._C.broadcast_tensor.restype = None
    
    Tensor._C.broadcast_tensor(tensor.tensor, src)

def allreduce_sum_tensor(tensor):

    Tensor._C.allreduce_sum_tensor.argtypes = [ctypes.POINTER(CTensor)]
    Tensor._C.allreduce_sum_tensor.restype = None
    
    Tensor._C.allreduce_sum_tensor(tensor.tensor)

def allreduce_mean_tensor(tensor):

    Tensor._C.allreduce_mean_tensor.argtypes = [ctypes.POINTER(CTensor)]
    Tensor._C.allreduce_mean_tensor.restype = None
    
    Tensor._C.allreduce_mean_tensor(tensor.tensor)
