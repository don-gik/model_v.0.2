import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    input: [B, 1, C, H, W] → squeeze → [B, C, H, W] → coord concat → [B, C+2, H, W]
    output: [B, D, H/2, W/2]
    """
    def __init__(self, in_channels=4, embed_dim=256, mid_dim=64):
        super().__init__()
        self.temporal = nn.Conv2d(in_channels, mid_dim, kernel_size=3, padding=1)
        self.spatial = nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=2, padding=1)

        self.dilated = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=2, dilation=2, groups=mid_dim),
            nn.BatchNorm2d(mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=1)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mid_dim * 2, mid_dim, kernel_size=1)
        )

        self.to_embed = nn.Conv2d(mid_dim, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def add_coord_channels(self, x):
        B, C, H, W = x.shape
        device = x.device
        i_coords = torch.linspace(-1, 1, steps=H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        j_coords = torch.linspace(-1, 1, steps=W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, i_coords, j_coords], dim=1)  # [B, C+2, H, W]

    def forward(self, x):
        x = x.squeeze(1)  # [B, C, H, W]
        x = self.add_coord_channels(x)  # → [B, C+2, H, W]

        x = self.temporal(x)
        x = self.spatial(x)

        residual = x
        x = self.dilated(x) + residual

        residual = x
        x = self.bottleneck(x) + residual

        x = self.to_embed(x)  # [B, D, H, W]
        B, D, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, D)
        x = self.norm(x).reshape(B, H, W, D).permute(0, 3, 1, 2)
        return x  # [B, D, H, W]
