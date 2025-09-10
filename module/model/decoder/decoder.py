import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

class ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(1, ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.GroupNorm(1, ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
    def forward(self, x):
        return x + self.block(x)
# decoder.py
class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels=4, hidden=64, num_res=2,
                 dropout=0.0, use_deconv=True):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, hidden, kernel_size=1)
        if use_deconv:
            self.up = nn.ConvTranspose2d(hidden, hidden, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(hidden, hidden, 3, padding=1),
            )
        self.res_stack = nn.Sequential(*[
            nn.Sequential(
                nn.GroupNorm(1, hidden), nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
                nn.GroupNorm(1, hidden), nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
            ) for _ in range(num_res)
        ])
        self.head = nn.Conv2d(hidden, out_channels, 3, padding=1)

        # 초기화
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        # 안전 가드
        if z.shape[1] != self.in_proj.in_channels:
            raise RuntimeError(f"Decoder in_channels mismatch: expected "
                               f"{self.in_proj.in_channels}, got {z.shape[1]}")
        h = self.in_proj(z)
        h = self.up(h)
        h = self.res_stack(h)
        return self.head(h)
