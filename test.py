import norch

W1 = norch.Tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], requires_grad=True)  # A has shape 10x1
X = norch.Tensor([[1, 2, 3, 4, 5]])  # B has shape 1x5
B1 = norch.Tensor([[1, 2, 3, 4, 5], [2, 1, 2, 3, 4], [3, 1,2,3, 4], [4, 1, 1, 1, 1,], [5,1,1,1,1], [6,1,1,1,1], [7,1,1,1,1], [8,1,1,1,1], [9,1,1,1,1], [10,1,1,1,1]], requires_grad=True)
B1 = norch.Tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], requires_grad=True)  # A has shape 10x1

W2 = norch.Tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], requires_grad=True).T
B2 = norch.Tensor([[1, 2, 3, 4, 5]], requires_grad=True)


Z1 = W1 @ X + B1

# Perform matrix multiplication

Z2 = W2 @ Z1 * B2
print(Z2.shape)

l = Z2.sum()
l.backward()
print(l)

print("Resulting matrix shape:", B1.grad)


"""import norch
import norch.nn as nn
import norch.optim as optim
import random
import math

random.seed(1)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        
        return out

device = "cpu"
epochs = 1

model = MyModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
loss_list = []

x_values = [[0., 1], [0., 1], [0., 1], [0., 1], [0., 1], [0., 1], [0., 1], [0., 1], [0., 1], [0., 1]]

y_true = []
for x in x_values:
    y_true.append([math.pow(math.sin(x[0]), 2), math.pow(math.sin(x[1]), 2)])

batch_size = 5


for epoch in range(epochs):
    for x, target in zip(x_values, y_true):
        x = norch.Tensor([x])
        target = norch.Tensor([target])

        x = x.to(device)
        target = target.to(device)

        outputs = model(x)

        loss = criterion(outputs, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('loss', loss, '\n\n')
        print('weight', model.fc1.weight, '\n\n')
        print('bias', model.fc1.bias, '\n\n')


        if math.isnan(loss[0]):
            print("Kkkkkkkkk")
            exit()

    #print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss[0]:.4f}')
    loss_list.append(loss[0])
"""