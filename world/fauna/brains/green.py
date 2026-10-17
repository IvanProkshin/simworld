import torch
import torch.nn.functional as F
import torch.nn as nn


class Green(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(22, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 2)

    def forward(self, x):
        r = F.gelu(self.fc1(x))
        r = F.gelu(self.fc2(r))
        r = torch.tanh(self.fc3(r))
        return r
