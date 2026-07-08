"""Encodeur CNN pour sprites 32x32 one-hot 16 canaux.

Architecture (CNN simple, justifiée par le brief : pas de ViT sur 32x32) :

  Input (16, 32, 32)
   -> Conv 3x3 stride 2 -> 64 ch  (16x16)
   -> Conv 3x3 stride 2 -> 128 ch (8x8)
   -> Conv 3x3 stride 2 -> 256 ch (4x4)
   -> Conv 3x3 stride 1 -> 256 ch (4x4)
   -> Global average pool         (256,)
   -> Linear -> 128
   -> L2 normalize                (128,)

GELU + GroupNorm (peu de samples par norm batch sur petits batchs, GN est plus
stable). On garde un "backbone" + "projector" distincts par convention SimCLR,
mais comme on n'a pas de tâche downstream supervisée à ce stade on expose
directement le vecteur normalisé.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int, groups_max: int = 16) -> nn.GroupNorm:
    g = min(groups_max, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm = _gn(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SpriteEncoder(nn.Module):
    def __init__(self, in_ch: int = 16, feat_dim: int = 128, widths=(64, 128, 256, 256)):
        super().__init__()
        c1, c2, c3, c4 = widths
        self.b1 = ConvBlock(in_ch, c1, stride=2)  # 32 -> 16
        self.b2 = ConvBlock(c1, c2, stride=2)     # 16 -> 8
        self.b3 = ConvBlock(c2, c3, stride=2)     # 8 -> 4
        self.b4 = ConvBlock(c3, c4, stride=1)     # 4 -> 4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(c4, c4),
            nn.GELU(),
            nn.Linear(c4, feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.b1(x)
        h = self.b2(h)
        h = self.b3(h)
        h = self.b4(h)
        h = self.pool(h).flatten(1)  # (B, c4)
        z = self.proj(h)             # (B, feat_dim)
        z = F.normalize(z, dim=1)    # L2 normalize
        return z

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
