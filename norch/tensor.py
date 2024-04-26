import ctypes

class CTensor(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('strides', ctypes.POINTER(ctypes.c_int)),
        ('shape', ctypes.POINTER(ctypes.c_int)),
        ('ndim', ctypes.c_int),
        ('size', ctypes.c_int),
    ]

class Tensor:
    _C = ctypes.CDLL("../build/libtensor.so")

    def __init__(self, data):
        data, shape = self.flatten(data)
          # Adjust the path to the shared library
        self.data = (ctypes.c_float * len(data))(*data)
        self.shape = shape
        self.ndim = len(shape)

        Tensor._C.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        Tensor._C.create_tensor.restype = ctypes.POINTER(CTensor)

        self.tensor = Tensor._C.create_tensor(
            self.data,
            (ctypes.c_int * len(shape))(*shape),
            ctypes.c_int(len(shape))
        )

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
        
        Tensor._C.get_item.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_int)]
        Tensor._C.get_item.restype = ctypes.c_float
                                           
        indices = (ctypes.c_int * len(indices))(*indices)
        value = Tensor._C.get_item(self.tensor, indices)  

        return value 

    def __str__(self):
        result = ""
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result += str(self[i, j]) + " "
            result += "\n"
        return result.strip()

    def __repr__(self):
        return self.__str__()
    
    def __add__(self, other):
        if self.shape != other.shape:
            raise ValueError("Tensors must have the same shape for addition")
        
        Tensor._C.add_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
        Tensor._C.add_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.add_tensor(self.tensor, other.tensor)

        result_data = [result_tensor_ptr.contents.data[i] for i in range(self.shape[0] * self.shape[1])]
        result_shape = [result_tensor_ptr.contents.shape[i] for i in range(result_tensor_ptr.contents.ndim)]

        return Tensor(result_data, result_shape)


if __name__ == "__main__":
    from tensor import Tensor
    import time

    ini = time.time()
    a = Tensor([[1, 2, 3], [1, 2, 3]])
    b = Tensor([[1, 2, 3], [1, 2, 3]])

    c = a + b
    fim = time.time()

    print(fim-ini)