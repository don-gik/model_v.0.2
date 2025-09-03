import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):  # t: [B] (LongTensor)
        if len(t.shape) == 1:
            t = t.float().unsqueeze(1)  # [B, 1]

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.mlp(emb)  # [B, dim]


class AdaGN(nn.Module):
    def __init__(self, num_channels, cond_channels=256, num_groups=16):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)
        self.cond_to_gamma_beta = nn.Conv2d(
            in_channels=num_channels, out_channels=num_channels * 2, kernel_size=1
        )

        self.needs_projection = cond_channels is not None and cond_channels != num_channels
        if self.needs_projection:
            self.cond_proj = nn.Conv2d(cond_channels, num_channels, 1)

    def forward(self, x, cond):  # x, cond: [B, C, H, W]
        x = self.norm(x)
        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(cond, size=x.shape[2:], mode='bilinear', align_corners=False)
        if self.needs_projection:
            cond = self.cond_proj(cond)
        gamma_beta = self.cond_to_gamma_beta(cond)  # [B, 2C, H, W]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return gamma * x + beta


class DiffusionResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.adagn1 = AdaGN(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.adagn2 = AdaGN(dim)
        self.activation = nn.SiLU()

    def forward(self, x, cond):
        residual = x
        x = self.activation(self.adagn1(self.conv1(x), cond))
        x = self.activation(self.adagn2(self.conv2(x), cond))
        return x + residual


class AstraEpsPredDiffuser(nn.Module):
    def __init__(self, in_channels=4, base_dim=128, time_dim=128):
        super().__init__()
        self.time_embed = TimestepEmbedding(time_dim)
        self.time_proj = nn.Linear(time_dim, base_dim)

        self.input_proj = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        self.resblock1 = DiffusionResBlock(base_dim)
        self.resblock2 = DiffusionResBlock(base_dim)
        self.resblock3 = DiffusionResBlock(base_dim)
        self.out_proj = nn.Conv2d(base_dim, in_channels, 3, padding=1)

    def forward(self, x_t, t, cond):
        """
        x_t: [B, C, H, W] — noised input (e.g. 4 channels)
        t:   [B] — diffusion timestep
        cond: [B, C, H, W] — ASTRA's last timestep output (z_cond)
        """
        t_feat = self.time_proj(self.time_embed(t))[:, :, None, None]  # [B, base_dim, 1, 1]
        x = self.input_proj(x_t) + t_feat
        x = self.resblock1(x, cond + t_feat)
        x = self.resblock2(x, cond + t_feat)
        x = self.resblock3(x, cond + t_feat)
        return self.out_proj(x)
