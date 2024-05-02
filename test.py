
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

    a = norch.Tensor([
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
        [7.890, 8.901, 5.91],
    ]])

    result = b @ a
    print(result)
    #c = result.sum()
    #c.backward()
    #print(a.grad)
    
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