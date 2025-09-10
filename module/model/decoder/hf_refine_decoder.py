import torch
import torch.nn as nn
import torch.nn.functional as F


class HighPass(nn.Module):
    """
    Depthwise Laplacian high-pass filter to extract edges/high-frequency content.
    """
    def __init__(self, channels: int):
        super().__init__()
        kernel = torch.tensor(
            [[0.0, -1.0, 0.0],
             [-1.0, 4.0, -1.0],
             [0.0, -1.0, 0.0]], dtype=torch.float32
        )
        weight = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=1, groups=self.channels)


class RefineBlock(nn.Module):
    def __init__(self, ch: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(1, ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.GroupNorm(1, ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class HFRefiner(nn.Module):
    """
    UNet-lite style high-frequency refiner producing a residual to add to base output.
    Inputs: concat of [base_out, highpass(base_out), cond_feat(optional)]
    """
    def __init__(self, in_ch: int, out_ch: int, width: int = 64, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Conv2d(in_ch, width, 3, padding=1)
        blocks = []
        dilations = [1, 2, 3][:max(1, depth)]
        for d in dilations:
            blocks.append(RefineBlock(width, dilation=d, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.Conv2d(width, out_ch, 3, padding=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.in_proj(x)
        h = self.blocks(h)
        return self.out_proj(h)


class FreqRefineDecoder(nn.Module):
    """
    Wraps a base spatial decoder with a high-frequency refinement head.
    - base_decoder: maps latent z [B,D,h,w] -> coarse [B,C,H,W]
    - refiner: predicts residual [B,C,H,W] from [base, highpass(base), cond(z)]
    """
    def __init__(self,
                 base_decoder: nn.Module,
                 in_channels: int,
                 out_channels: int = 4,
                 cond_z: bool = True,
                 refine_width: int = 64,
                 refine_depth: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base_decoder
        self.out_channels = out_channels
        self.cond_z = cond_z

        # Project encoder feature z to out_channels and upsample to image resolution
        if self.cond_z:
            self.cond_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # High-pass filter for base output
        self.hpf = HighPass(out_channels)

        in_ch = out_channels + out_channels + (out_channels if self.cond_z else 0)
        self.refiner = HFRefiner(in_ch, out_channels, width=refine_width, depth=refine_depth, dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Base reconstruction
        base = self.base(z)  # [B,C,H,W]
        hp = self.hpf(base)
        if self.cond_z:
            B, C, H, W = base.shape
            cond = self.cond_proj(z)
            if cond.shape[-2:] != (H, W):
                cond = F.interpolate(cond, size=(H, W), mode='bilinear', align_corners=False)
            ref_in = torch.cat([base, hp, cond], dim=1)
        else:
            ref_in = torch.cat([base, hp], dim=1)
        resid = self.refiner(ref_in)
        return base + resid


class SwinFreqRefineDecoder(nn.Module):
    """
    Time-aware high-frequency refinement for Swin features.
    - base_decoder: module mapping [B,T,D,h,w] -> [B,T,C,H,W] (e.g., SwinDecoder)
    - Uses HighPass and HFRefiner per timestep with optional conditioning on z.
    """
    def __init__(self,
                 base_decoder: nn.Module,
                 in_channels: int,
                 out_channels: int = 4,
                 cond_z: bool = True,
                 refine_width: int = 64,
                 refine_depth: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base_decoder
        self.out_channels = out_channels
        self.cond_z = cond_z
        if self.cond_z:
            self.cond_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.hpf = HighPass(out_channels)
        in_ch = out_channels + out_channels + (out_channels if self.cond_z else 0)
        self.refiner = HFRefiner(in_ch, out_channels, width=refine_width, depth=refine_depth, dropout=dropout)

    def forward(self, z_btDhw: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        # z_btDhw: [B,T,D,h,w]
        B, T, D, h, w = z_btDhw.shape
        H, W = out_size
        # Base outputs per timestep
        y_btchw = self.base(z_btDhw, out_size)  # [B,T,C,H,W]
        # Flatten time for efficient refinement
        y = y_btchw.reshape(B * T, self.out_channels, H, W)
        hp = self.hpf(y)
        if self.cond_z:
            z_flat = z_btDhw.reshape(B * T, D, h, w)
            cond = self.cond_proj(z_flat)
            if cond.shape[-2:] != (H, W):
                cond = F.interpolate(cond, size=(H, W), mode='bilinear', align_corners=False)
            ref_in = torch.cat([y, hp, cond], dim=1)
        else:
            ref_in = torch.cat([y, hp], dim=1)
        resid = self.refiner(ref_in)
        y_ref = (y + resid).view(B, T, self.out_channels, H, W)
        return y_ref
