import ctypes

class CTensor(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('strides', ctypes.POINTER(ctypes.c_int)),
        ('shape', ctypes.POINTER(ctypes.c_int)),
        ('ndim', ctypes.c_int),
        ('size', ctypes.c_int),
        ('device', ctypes.c_char_p)
    ]

class Tensor:
    _C = ctypes.CDLL("../build/libtensor.so")

    def __init__(self, data=None, device="cpu"):

        if data != None:
            data, shape = self.flatten(data)
            # Adjust the path to the shared library
            self.data_ctype = (ctypes.c_float * len(data))(*data)
            self.shape_ctype = (ctypes.c_int * len(shape))(*shape)
            self.ndim_ctype = ctypes.c_int(len(shape))
            self.device_ctype = device.encode('utf-8')


            self.shape = shape
            self.ndim = len(shape)
            self.device = device

            Tensor._C.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p]
            Tensor._C.create_tensor.restype = ctypes.POINTER(CTensor)
            

            self.tensor = Tensor._C.create_tensor(
                self.data_ctype,
                self.shape_ctype,
                self.ndim_ctype,
                self.device_ctype
            )
        
        else:
            self.tensor = None,
            self.shape = None,
            self.ndim = None,
            self.device = device

    def flatten(self, nested_list):
        def flatten_recursively(nested_list):
            flat_data = []
            shape = []
            if isinstance(nested_list, list):
                for sublist in nested_list:
                    inner_data, inner_shape = flatten_recursively(sublist)
                    flat_data.extend(inner_data)
                shape.append(len(nested_list))
                shape.extend(inner_shape)
            else:
                flat_data.append(nested_list)
            return flat_data, shape
        
        flat_data, shape = flatten_recursively(nested_list)
        return flat_data, shape
    
    def reshape(self, new_shape):

        new_shape_ctype = (ctypes.c_int * len(new_shape))(*new_shape)
        new_ndim_ctype = ctypes.c_int(len(new_shape))
        
        Tensor._C.reshape_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        Tensor._C.reshape_tensor.restype = None
        Tensor._C.reshape_tensor(self.tensor, new_shape_ctype, new_ndim_ctype)   

        self.shape = new_shape
        self.ndim = len(new_shape)

        return self
    
    def to(self, device):
        self.device = device
        self.device_ctype = device.encode('utf-8')

        Tensor._C.to_device.argtypes = [ctypes.POINTER(CTensor), ctypes.c_char_p]
        Tensor._C.to_device.restype = None
        Tensor._C.to_device(self.tensor, self.device_ctype)

    def __getitem__(self, indices):
        if len(indices) != self.ndim:
            raise ValueError("Number of indices must match the number of dimensions")
        
        Tensor._C.get_item.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_int)]
        Tensor._C.get_item.restype = ctypes.c_float
                                           
        indices = (ctypes.c_int * len(indices))(*indices)
        value = Tensor._C.get_item(self.tensor, indices)  

        return value 
    
    def __str__(self):
        def print_recursively(tensor, depth, index):
            if depth == tensor.ndim - 1:
                result = ""
                for i in range(tensor.shape[-1]):
                    index[-1] = i
                    result += str(tensor[tuple(index)]) + ", "
                return result.strip()
            else:
                result = ""
                if depth > 0:
                    result += "\n" + " " * ((depth - 1) * 4)
                for i in range(tensor.shape[depth]):
                    index[depth] = i
                    result += "["
                    result += print_recursively(tensor, depth + 1, index) + "],"
                    if i < tensor.shape[depth] - 1:
                        result += "\n" + " " * (depth * 4)
                return result.strip(",")

        index = [0] * self.ndim
        result = "tensor(["
        result += print_recursively(self, 0, index)
        result += "])"
        return result

    def __repr__(self):
        return self.__str__()
    
    def __add__(self, other):
        if self.shape != other.shape:
            raise ValueError("Tensors must have the same shape for addition")
        
        Tensor._C.add_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
        Tensor._C.add_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.add_tensor(self.tensor, other.tensor)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape.copy()
        result_data.ndim = self.ndim

        return result_data
    
    def __sub__(self, other):
        if self.shape != other.shape:
            raise ValueError("Tensors must have the same shape for subtraction")
        
        Tensor._C.sub_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
        Tensor._C.sub_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.sub_tensor(self.tensor, other.tensor)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape.copy()
        result_data.ndim = self.ndim

        return result_data
    
    def __mul__(self, other):
        if self.shape != other.shape:
            raise ValueError("Tensors must have the same shape for element-wise multiplication")
        
        Tensor._C.elementwise_mul_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
        Tensor._C.elementwise_mul_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.elementwise_mul_tensor(self.tensor, other.tensor)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape.copy()
        result_data.ndim = self.ndim

        return result_data
    
    def __matmul__(self, other):
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("Matrix multiplication requires 2D tensors")

        if self.shape[1] != other.shape[0]:
            raise ValueError("Incompatible shapes for matrix multiplication")

        Tensor._C.matmul_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
        Tensor._C.matmul_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.matmul_tensor(self.tensor, other.tensor)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = [self.shape[0], other.shape[1]]
        result_data.ndim = 2

        return result_data

    def __pow__(self, power):
        
        power = ctypes.c_float(power)

        Tensor._C.pow_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float]
        Tensor._C.pow_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.pow_tensor(self.tensor, power)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape.copy()
        result_data.ndim = 2

        return result_data


def matrix_sum(matrix1, matrix2):
    # Check if the matrices can be multiplied
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Matrices cannot be multiplied. Inner dimensions must match.")

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]

    # Perform matrix multiplication
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            result[i][j] += matrix1[i][i] + matrix2[i][j]

    return result

if __name__ == "__main__":
    from tensor import Tensor
    import time
    import random
    import numpy as np

    
    a = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    print(a ** 4)
    
    """#print(a)
    N = 1000
    a = Tensor([[random.uniform(0, 1) for _ in range(N)] for _ in range(N)])
    b = Tensor([[random.uniform(0, 1) for _ in range(N)] for _ in range(N)])
    #b = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    #a = Tensor([[[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]]])
    #b = Tensor([[[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]]])
    ini = time.time()
    c = a + b

    print("\n#####2######")
    
    fim = time.time()

    print(fim-ini)

    print("\n\n")

    
    a = [[random.uniform(0, 1) for _ in range(N)] for _ in range(N)]
    b = [[random.uniform(0, 1) for _ in range(N)] for _ in range(N)]
    ini = time.time()
    result_matrix = matrix_sum(a, b)
    fim = time.time()
    print(fim-ini)


    print("\n\n")

    ini = time.time()
    a = np.random.rand(N, N)
    b = np.random.rand(N, N)
    result_matrix = a + b
    fim = time.time()
    print(fim-ini)

"""