import norch
import norch.nn as nn
import norch.optim as optim
from norch.norchvision import transforms as T
import random
random.seed(1)

from memory_profiler import profile

@profile
def main():

    BATCH_SIZE = 32
    device = "cpu"
    epochs = 10

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Reshape([-1, 784, 1])
        ]
    )

    target_transform = T.Compose(
        [
            T.ToTensor()
        ]
    )


    print("Loading data")
    train_data, test_data = norch.norchvision.datasets.MNIST.splits(transform=transform, target_transform=target_transform)
    train_loader = norch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.fc1 = nn.Linear(784, 30)
            self.sigmoid1 = nn.Sigmoid()
            self.fc2 = nn.Linear(30, 10)
            self.sigmoid2 = nn.Sigmoid()

        def forward(self, x):
            out = self.fc1(x)
            out = self.sigmoid1(out)
            out = self.fc2(out)
            out = self.sigmoid2(out)
            
            return out

    print("Creating model")
    model = MyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_list = []

    print("Starting training")
    for epoch in range(epochs): 

        avg_loss = 0  
        num_steps = 0  
        
        for idx, batch in enumerate(train_loader):

            if idx % 300 == 0 and idx > 0:
                print(f"Epoch: {epoch}/{epochs} - Step: {idx} / {len(train_loader)}")
                break

            inputs, target = batch

            inputs = inputs.to(device)
            target = target.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, target)
            
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()
        
            avg_loss += loss[0]
            num_steps += 1

        break
        avg_loss = avg_loss / num_steps
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')
        loss_list.append(avg_loss)
        


if __name__ == "__main__":
    main()

