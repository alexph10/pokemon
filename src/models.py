"""Conditional GAN baseline.

Generator: DCGAN-style ConvTranspose stack, class embedding added to z.
Discriminator: spectral-norm Conv stack + projection conditioning.
Supports img_size in {64, 128, 256}; output is RGB tanh in [-1, 1].
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN

_UPS = {64: 4, 128: 5, 256: 6}


class Generator(nn.Module):
    def __init__(self, z_dim: int = 128, n_classes: int = 10,
                 ch: int = 64, img_size: int = 128):
        super().__init__()
        assert img_size in _UPS, f"img_size must be in {list(_UPS)}"
        self.z_dim = z_dim
        self.embed = nn.Embedding(n_classes, z_dim)

        ups = _UPS[img_size]
        base = ch * (2 ** (ups - 1))  # widest channel count at 4x4

        # Project z -> 4x4 feature map.
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(z_dim, base, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(True),
        ]
        cur = base
        # Upsample ups-1 times, halving channels (floored at ch).
        for _ in range(ups - 1):
            nxt = max(cur // 2, ch)
            layers += [
                nn.ConvTranspose2d(cur, nxt, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nxt), nn.ReLU(True),
            ]
            cur = nxt
        # Final upsample to img_size with RGB output.
        layers += [nn.ConvTranspose2d(cur, 3, 4, 2, 1), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Additive class conditioning on z.
        zc = z + self.embed(y)
        return self.net(zc.view(-1, self.z_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, n_classes: int = 10, ch: int = 64,
                 img_size: int = 128):
        super().__init__()
        assert img_size in _UPS
        downs = _UPS[img_size]

        layers: list[nn.Module] = [
            SN(nn.Conv2d(3, ch, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),
        ]
        cur = ch
        for _ in range(downs - 1):
            nxt = min(cur * 2, ch * 16)
            layers += [
                SN(nn.Conv2d(cur, nxt, 4, 2, 1)),
                nn.LeakyReLU(0.2, True),
            ]
            cur = nxt
        self.features = nn.Sequential(*layers)

        # Projection discriminator: real-valued logit + <embed(y), pooled h>.
        self.fc = SN(nn.Linear(cur, 1))
        self.embed = SN(nn.Embedding(n_classes, cur))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.features(x).sum(dim=(2, 3))   # global sum pool -> (B, C)
        out = self.fc(h).squeeze(1)            # unconditional logit
        out = out + (self.embed(y) * h).sum(1) # + class projection
        return out
