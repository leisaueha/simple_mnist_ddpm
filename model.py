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


class ClassConditioning(nn.Module):
    """Add a class or learned null-class embedding to a time embedding."""

    def __init__(self, num_classes, time_dim):
        super().__init__()
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes + 1, time_dim)

    def forward(self, time_embedding, labels):
        batch_size = time_embedding.size(0)
        if labels is None:
            labels = torch.full(
                (batch_size,),
                self.num_classes,
                device=time_embedding.device,
                dtype=torch.long,
            )
        else:
            labels = labels.to(device=time_embedding.device, dtype=torch.long)
            if labels.ndim != 1 or labels.size(0) != batch_size:
                raise ValueError("labels must have shape [batch_size]")
            if ((labels < -1) | (labels >= self.num_classes)).any():
                raise ValueError(
                    f"labels must be in [0, {self.num_classes - 1}] or -1"
                )
            labels = torch.where(
                labels == -1,
                torch.full_like(labels, self.num_classes),
                labels,
            )

        return time_embedding + self.embedding(labels)


class SimpleUNet(nn.Module):
    def __init__(self, base_channels=32, time_dim=128, num_classes=None):
        super().__init__()
        self.time_dim = time_dim
        self.num_classes = num_classes
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        if num_classes is not None:
            # The final entry is the learned null class used for label dropout
            # during training and the unconditional CFG prediction at sampling.
            self.class_conditioning = ClassConditioning(num_classes, time_dim)

        self.down1 = Block(1, base_channels, time_dim)
        self.down2 = Block(base_channels, base_channels * 2, time_dim)
        self.middle = Block(base_channels * 2, base_channels * 2, time_dim)

        self.up1 = Block(base_channels * 4, base_channels, time_dim)
        self.up2 = Block(base_channels * 2, base_channels, time_dim)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.output = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x, t, labels=None):
        time_embedding = self.time_mlp(timestep_embedding(t, self.time_dim))
        if self.num_classes is not None:
            time_embedding = self.class_conditioning(time_embedding, labels)
        elif labels is not None:
            raise ValueError("this model was created without class conditioning")

        x1 = self.down1(x, time_embedding)          # 28 x 28
        x2 = self.down2(self.pool(x1), time_embedding)  # 14 x 14
        x3 = self.middle(self.pool(x2), time_embedding) # 7 x 7

        x = self.upsample(x3)                      # 14 x 14
        x = self.up1(torch.cat([x, x2], dim=1), time_embedding)

        x = self.upsample(x)                       # 28 x 28
        x = self.up2(torch.cat([x, x1], dim=1), time_embedding)

        return self.output(x)
