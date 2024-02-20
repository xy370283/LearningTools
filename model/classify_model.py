import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    """
        just a simple net for classification, only for demo
    """

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x