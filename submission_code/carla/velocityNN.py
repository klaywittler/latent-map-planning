import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

class velocityNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(64,512),
            nn.ReLU(True),
            nn.Dropout(0.2))
        
        self.vel = nn.Linear(512,6)
        self.steer = nn.Linear(512,11)

    def forward(self, x):
        h = self.fc(x)
        vel = self.vel(h)
        steer = self.steer(h)
        # vel = F.softmax(vel, dim=6)
        # vel = F.softmax(steer, dim=11)
        return vel.unsqueeze(0) , steer.unsqueeze(0)