import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

class CustomizedConv2d(nn.Module):
    def __init__(self, hidden_dims, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(1, hidden_dims, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x).squeeze(-2)
    
class ConvolutionEncoder(nn.Module):
    def __init__(self, hidden_dims, z_dims):
        super().__init__()
        self.encoder = nn.Sequential(
            CustomizedConv2d(hidden_dims=hidden_dims, kernel_size=(4, 5)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dims, out_channels=hidden_dims, kernel_size=8),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dims, out_channels=hidden_dims, kernel_size=8),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dims, out_channels=hidden_dims, kernel_size=8),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dims, out_channels=hidden_dims, kernel_size=8),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dims, out_channels=z_dims, kernel_size=7),
        )

    def forward(self, x):
        return self.encoder(x)
    
class MLPEncoder(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, x):
        pass