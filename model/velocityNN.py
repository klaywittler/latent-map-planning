import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

class velocityNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64,256),
            nn.ReLU(True),
            nn.Linear(256,256))

    def forward(self, x):
        h = self.fc(x)
        vel = F.softmax(h, dim=6)
        steer = F.softmax(h, dim=11)
        return vel , steer