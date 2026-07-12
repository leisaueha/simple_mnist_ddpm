import math

import torch
from torch import nn


def timestep_embedding(t, dim):
    """Sinusoidal timestep embeddings."""
    half = dim // 2
    scale = math.log(10000) / max(half - 1, 1)
    frequencies = torch.exp(
        -scale * torch.arange(half, device=t.device, dtype=torch.float32)
    )
    embedding = t.float()[:, None] * frequencies[None, :]
    embedding = torch.cat([embedding.sin(), embedding.cos()], dim=1)

    if dim % 2 == 1:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=1
        )
    return embedding


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self, x, time_embedding):
        x = self.act(self.norm1(self.conv1(x)))
        x = x + self.time_proj(time_embedding)[:, :, None, None]
        x = self.act(self.norm2(self.conv2(x)))
        return x


class SimpleUNet(nn.Module):
    def __init__(self, base_channels=32, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.down1 = Block(1, base_channels, time_dim)
        self.down2 = Block(base_channels, base_channels * 2, time_dim)
        self.middle = Block(base_channels * 2, base_channels * 2, time_dim)

        self.up1 = Block(base_channels * 4, base_channels, time_dim)
        self.up2 = Block(base_channels * 2, base_channels, time_dim)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.output = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x, t):
        time_embedding = self.time_mlp(timestep_embedding(t, self.time_dim))

        x1 = self.down1(x, time_embedding)          # 28 x 28
        x2 = self.down2(self.pool(x1), time_embedding)  # 14 x 14
        x3 = self.middle(self.pool(x2), time_embedding) # 7 x 7

        x = self.upsample(x3)                      # 14 x 14
        x = self.up1(torch.cat([x, x2], dim=1), time_embedding)

        x = self.upsample(x)                       # 28 x 28
        x = self.up2(torch.cat([x, x1], dim=1), time_embedding)

        return self.output(x)
