import norch
from norch.utils.data.dataloader import Dataloader
import norch
import norch.nn as nn
import norch.optim as optim
import random
random.seed(1)

a = norch.Tensor([1, 2, 3])
b = 3
for i in range(a.numel):
    print(a.tensor.contents.data[i])

print(a)

"""

to_tensor = lambda x: norch.Tensor(x)
reshape = lambda x: x.reshape([-1, 784])
transform = lambda x: reshape(to_tensor(x))
target_transform = lambda x: to_tensor(x)

train_data, test_data = norch.datasets.MNIST.splits(transform=transform, target_transform=target_transform)
sample, _ = train_data[0]

BATCH_SIZE = 100

train_loader = Dataloader(train_data, batch_size = BATCH_SIZE)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(784, 10)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        
        return out

device = "cpu"
epochs = 10

model = MyModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
loss_list = []

for epoch in range(epochs):    
    for idx, batch in enumerate(train_loader):
        x, target = batch

        x = x.T
        target = target.T 

        x = x.to(device)
        target = target.to(device)

        outputs = model(x)

        loss = criterion(outputs, target)
        
        optimizer.zero_grad()
        loss.backward()
        print('loss_antes', loss)

        print('f1 antes', model.fc1.bias)
        print('f1 grad_antes', model.fc1.bias.grad)
        print('f2 antes', model.fc2.bias)
        print('f2 grad_antes', model.fc2.bias.grad)

        optimizer.step()
        print('\n')

        print('f1 depois', model.fc1.bias)
        print('f2 depois', model.fc2.bias)
        print('\n\n')

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss[0]:.4f}')
    loss_list.append(loss[0])"""