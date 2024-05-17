import norch

W1 = norch.Tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], requires_grad=True)  # A has shape 10x1
X = norch.Tensor([[1, 2, 3, 4, 5]])  # B has shape 1x5
B1 = norch.Tensor([[1, 2, 3, 4, 5], [2, 1, 2, 3, 4], [3, 1,2,3, 4], [4, 1, 1, 1, 1,], [5,1,1,1,1], [6,1,1,1,1], [7,1,1,1,1], [8,1,1,1,1], [9,1,1,1,1], [10,1,1,1,1]], requires_grad=True)


W2 = norch.Tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], requires_grad=True).T
B2 = norch.Tensor([[1, 2, 3, 4, 5]], requires_grad=True)


Z1 = W1 @ X + B1

# Perform matrix multiplication

Z2 = W2 @ Z1 + B2
print(Z2.shape)

l = Z2.sum()
l.backward()
#print(l)

print("Resulting matrix shape:", W1.grad)

