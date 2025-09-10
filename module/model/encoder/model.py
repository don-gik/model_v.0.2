import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels=4, embed_dim=64, mid_dim=16, use_coords=False):
        super().__init__()
        self.use_coords = use_coords
        self.embed_dim = embed_dim
        in_ch = in_channels + (2 if use_coords else 0)
        # Use augmented channel count when coords are enabled
        self.temporal = nn.Conv2d(in_ch, mid_dim, kernel_size=3, padding=1)
        self.spatial  = nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=2, padding=1)
        self.dilated  = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, 3, padding=2, dilation=2, groups=mid_dim),
            nn.BatchNorm2d(mid_dim), nn.GELU(),
            nn.Conv2d(mid_dim, mid_dim, 1)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim*2, 1), nn.GELU(), nn.Conv2d(mid_dim*2, mid_dim, 1)
        )
        self.to_embed = nn.Conv2d(mid_dim, embed_dim, 1)
        self.norm = nn.LayerNorm(embed_dim)

    def add_coord_channels(self, x):
        B, C, H, W = x.shape
        dev = x.device
        i = torch.linspace(-1, 1, H, device=dev).view(1,1,H,1).expand(B,1,H,W)
        j = torch.linspace(-1, 1, W, device=dev).view(1,1,1,W).expand(B,1,H,W)
        return torch.cat([x, i, j], dim=1)

    def forward(self, x):
        x = x.squeeze(1)                           # [B, C, H, W]
        if self.use_coords: x = self.add_coord_channels(x)  # 입력 채널만 +2
        x = self.temporal(x); x = self.spatial(x)
        x = self.dilated(x) + x
        x = self.bottleneck(x) + x
        x = self.to_embed(x)                       # [B, 4, H/2, W/2]
        B, D, H, W = x.shape
        x = x.permute(0,2,3,1).reshape(B*H*W, D)
        x = self.norm(x).reshape(B,H,W,D).permute(0,3,1,2)
        return x

    @property
    def out_dim(self):
        return self.embed_dim
