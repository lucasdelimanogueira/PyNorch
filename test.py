
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
    import norch
    import time
    import random
    import numpy as np

    """a = norch.Tensor([
        [[1.234, 2.123], [3.635, 4.456], [5.678, 6.789]],
        [[7.890, 8.901], [9.012, 1.234], [2.345, 3.456]],
        [[4.567, 5.678], [6.789, 7.890], [8.901, 9.012]],
        [[1.234, 2.345], [3.456, 4.567], [5.678, 6.789]],
        [[7.890, 8.901], [9.012, 1.234], [2.345, 3.456]]
    ], requires_grad=True)

    b = norch.Tensor([[
        [1.234, 2.123, 1.5],
        [5.678, 6.789, 1.293],
        [3.635, 4.456, 1.0202],
        [7.890, 8.901, 1.91],
    ],[
        [1.234, 2.123, 1.5],
        [5.678, 6.789, 1.293],
        [3.635, 4.456, 1.0202],
        [7.890, 8.901, 1.91],
    ],[
        [1.234, 2.123, 1.5],
        [5.678, 6.789, 1.293],
        [3.635, 4.456, 1.0202],
        [7.890, 8.901, 1.91],
    ],[
        [1.234, 2.123, 1.5],
        [5.678, 6.789, 1.293],
        [3.635, 4.456, 1.0202],
        [7.890, 8.901, 1.91],
    ],[
        [1.234, 2.123, 1.5],
        [5.678, 6.789, 1.293],
        [3.635, 4.456, 1.0202],
        [7.890, 8.901, 1.91],
    ]])

    #print(a.T)
    #print(a.T.shape)
    #[5, 3, 2] [4, 3] [5, 4, 2]
    #print(a.shape, b.shape)
    #b = norch.Tensor([
    #    [1.234, 2.123, 1.5]])
    result = b @ a
    result = result.sum()
    result.backward()

    print(a.grad)"""

    #### testar transpose axes!!!! make it contiguous

    tensor1 = norch.Tensor([[[1, 2], [3, 4], [5, 6]],
                        [[7, 8], [9, 10], [11, 12]],
                        [[13, 14], [15, 16], [17, 18]],
                        [[19, 20], [21, 22], [23, 24]],
                        [[25, 26], [27, 28], [29, 30]]], requires_grad=True)

    # Reshape tensor1 to 2x3x5
    reshaped_tensor = tensor1.transpose(1, 0)

    # Create a 5x4 tensor
    tensor2 = norch.Tensor([[1, 2, 3],
                            [5, 6, 7],
                            [9, 10, 11],
                            [13, 14, 15],
                            [17, 18, 19]])

    tensor2 = tensor2.transpose(1,0)
    

    # Multiply reshaped_tensor by tensor2
    result = tensor2 @ reshaped_tensor

    result = result.sum()
    result.backward()
    print(tensor1.grad)

    #print(a.shape, b.shape, result.shape)
    #c = result.sum()
    #c.backward()
    #print(a.grad)
    #print(a.transpose(2,1))
    
    #a = norch.Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])#.to("cuda")
    #b = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])#.to("cuda")
    #c = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])#.to("cuda")

    #d = b-c

    """a = norch.Tensor([[1, 2], [1, 2], [1, 2]], requires_grad=True)#.to("cuda")
    b = norch.Tensor([[1, 400, 3], [1, 2, 3]], requires_grad=True)

    c = (a@b).reshape([9])
    d = c.sum()
    d.backward()

    print(a.grad)"""
    
    """#print(a)
    N = 10
    a = norch.Tensor([[1 for _ in range(N)] for _ in range(N)])
    #b = norch.Tensor([[random.uniform(0, 1) for _ in range(N)] for _ in range(N)])
    #b = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    #a = Tensor([[[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]]])
    #b = Tensor([[[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]]])
    ini = time.time()
    c = a.sum()

    print("\n#####2######")
    
    fim = time.time()

    print(fim-ini)
    print(c)
    print("\n\n")

    """
    """
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