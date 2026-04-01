# model.py

import torch.nn as nn

class BasicNet(nn.Module):

    def __init__(self):
        super(BasicNet, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
           
        )

    def forward(self, x):
        return self.network(x)