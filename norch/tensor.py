import os
import sys
import ctypes
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'build'))
sys.path.append(build_dir)
import norch_C

class Tensor:
    def __init__(self, data=None):

        if data is not None:
            data, shape = self.flatten(data)
            self.shape = shape
            self.ndim = len(shape)

            self.tensor = norch_C.create_tensor(
                data, 
                self.shape, 
                self.ndim
            )

        else:
            self.shape = None
            self.ndim = 0
            self.tensor = None

    def flatten(self, nested_list):
        flat_data = []
        shape = [len(nested_list), len(nested_list[0])]
        for sublist in nested_list:
            for item in sublist:
                flat_data.append(item)
        return flat_data, shape
    
    def __getitem__(self, indices):
        if len(indices) != self.ndim:
            raise ValueError("Number of indices must match the number of dimensions")
        
        value = norch_C.get_item(self.tensor, indices)  

        return value
    
    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise ValueError("Can only add CTensor objects")

        if self.ndim != other.ndim:
            raise ValueError("Number of dimensions must match for addition")

        result_tensor_ptr = norch.add(self.tensor, other.tensor)

        new_tensor = Tensor()
        new_tensor.tensor = result_tensor_ptr
        new_tensor.shape = self.shape
        new_tensor.ndim = self.ndim
        
        return new_tensor

    def __str__(self):
        if self.tensor is None:
            return "Empty Tensor"
        else:
            result = ""
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result += str(self[i, j]) + " "
                result += "\n"
            return result.strip()

    def __repr__(self):
        return self.__str__()

if __name__ == '__main__':
    from tensor import Tensor
    import time

    ini = time.time()
    a = Tensor([1000000*[1, 2, 3]])
    fim = time.time()

    print(fim-ini)