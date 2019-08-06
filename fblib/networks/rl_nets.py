import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedPolicy(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(FullyConnectedPolicy, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = F.softmax(self.model(x), dim=1)
        return x
