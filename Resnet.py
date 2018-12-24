import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(outchannel)
                )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(outchannel)
                    )
    
    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x = self.left(x)
        x += x_shortcut
        x = F.relu(x)
        return x
        