import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import dataclass
import typing

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
        x = x.unsqueeze(1)
        return self.encoder(x)
    

#==================encoder-only Transformer Encoder======================
@dataclass
class TransformerEncoderConfig:
    block_size: int = 30
    n_layer: int = 3
    n_head: int = 8 # n_embd must be divisible by n_head
    n_embd: int = 64
    n_ltnt: int = 32 # latent dims
    dropout: float = 0.0
    device: str = 'cpu'

class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, config: TransformerEncoderConfig):
        super().__init__()
        self.config = config
        head_size = config.n_embd // config.n_head
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, _, T, C = x.shape # (B, 4, L, C)
        k = self.key(x) # (B, 4, L, head_size)
        q = self.query(x) # (B, 4, L, head_size)
        # compute attention scores
        att = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # (B, 4, L, L)
        att = F.softmax(att, dim=-1) # (B, 4, L, L)
        att = self.dropout(att)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, 4, L, head_size)
        out = att @ v # (B, 4, L, head_size)
        return out
    
class MultiHeadBidirectionalAttention(nn.Module):
    def __init__(self, config: TransformerEncoderConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, 4, L, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, config: TransformerEncoderConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = self.fc(x)
        out = self.gelu(out)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerEncoderConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadBidirectionalAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.fdfw = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fdfw(self.ln2(x))
        return x

class FloatEmbedding(nn.Module):
    def __init__(self, config: TransformerEncoderConfig):
        super().__init__()
        self.config = config
        self.freqs = nn.Parameter(torch.rand(config.n_embd//2))

    def forward(self, x):
        x = x.unsqueeze(-1).expand(-1, -1, -1, self.config.n_embd) # (B, Ty, L, C)
        # cos components
        cos_components = x[:, :, :, 0::2] * (2 * torch.pi * self.freqs)
        cos_components = torch.cos(cos_components)
        # sin components
        sin_components = x[:, :, :, 1::2] * (2 * torch.pi * self.freqs)
        sin_components = torch.sin(sin_components)
        return torch.concat([cos_components, sin_components], dim=-1)

class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerEncoderConfig):
        super().__init__()
        self.config = config
        self.repr_tkn = nn.Parameter(torch.zeros(1, 4, 1, config.n_embd)) # (1, Ty, 1, C)
        self.intensity_embd = FloatEmbedding(config)
        self.pos_embd = nn.Embedding(config.block_size + 1, config.n_embd)
        self.h = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.fc_f = nn.Linear(config.n_embd, config.n_ltnt // 4) #(C, Z / Ty)

    def forward(self, x):
        B, Ty, L = x.shape
        x = self.intensity_embd(x) # (B, Ty, L, C)
        x = torch.cat([x, self.repr_tkn.expand(B, -1, -1, -1)], dim=-2) # (B, Ty, L + 1, C)
        pos = torch.arange(0, L + 1, dtype=torch.long, device=self.config.device)
        x = x + self.pos_embd(pos)
        for layer in self.h:
            x = layer(x)
        x = self.ln_f(x)
        z = self.fc_f(x) # (B, Ty, L + 1, Z / Ty)
        out = z[:, :, -1, :].reshape(B, self.config.n_ltnt) # (B, Z)
        return out 
    


