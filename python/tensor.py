import ctypes

class CTensor(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('strides', ctypes.POINTER(ctypes.c_int)),
        ('shape', ctypes.POINTER(ctypes.c_int)),
        ('ndim', ctypes.c_int)
    ]

class Tensor:
    def __init__(self, data):
        data, shape = self.flatten(data)
        print(data, shape)
        self.lib = ctypes.CDLL("../build/libtensor.so")  # Adjust the path to the shared library
        self.data = (ctypes.c_float * len(data))(*data)
        self.shape = shape
        self.ndim = len(shape)

        self.lib.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.create_tensor.restype = ctypes.POINTER(CTensor)

        self.tensor = self.lib.create_tensor(
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
        
        self.lib.get_item.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_int)]
        self.lib.get_item.restype = ctypes.c_float
                                           
        indices = (ctypes.c_int * len(indices))(*indices)
        value = self.lib.get_item(self.tensor, indices)  

        return value

    def shape(self):
        return f"Tensor(shape={self.shape})"

    def __str__(self):
        result = ""
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result += str(self[i, j]) + " "
            result += "\n"
        return result.strip()

    def __repr__(self):
        return self.__str__()
