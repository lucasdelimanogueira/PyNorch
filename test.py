import norch
import norch.nn as nn
import norch.optim as optim
from norch.utils.data.dataloader import Dataloader
from norch.norchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(1)

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

model = MyModel().to("cuda")