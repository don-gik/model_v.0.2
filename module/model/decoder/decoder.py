import torch
import torch.nn as nn
import torch.nn.functional as F



class UpSample(nn.Module):
    def __init__(self,
                 channels : int
                 ):
        
        super().__init__()
        
        self.projection = nn.Conv3d(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 1
        )
    
    def forward(self, x):
        y = F.interpolate(
            input = x,
            scale_factor = (1, 2, 2),
            mode = "trilinear",
            align_corners = False
        )

        return self.projection(y)


class Refine3D(nn.Module):
    def __init__(self,
                 channels : int,
                 drop : float = 0.0
                 ):
        
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels = channels,
                out_channels = channels * 4,
                kernel_size = 1
            ),
            nn.GELU(),

            nn.Conv3d(
                in_channels = channels * 4,
                out_channels = channels * 4,
                kernel_size = (1, 3, 3),
                padding = "same",
                groups = channels
            ),
            nn.GELU(),

            nn.Conv3d(
                in_channels = channels * 4,
                out_channels = channels,
                kernel_size = 1
            ),
            nn.Dropout(drop)
        )
        nn.init.zeros_(self.block[-2].weight)
        if self.block[-2].bias is not None:
            nn.init.zeros_(self.block[-2].bias)
    
    def forward(self, x):
        return self.block(x)


class RefineBlockDecoder(nn.Module):
    def __init__(self,
                 channels : int,
                 outChannels : int,
                 depth : int = 3,
                 drop : float = 0.0
                 ):
        
        super().__init__()

        self.upsample = UpSample(channels = channels)
        self.blocks = nn.ModuleList([Refine3D(
            channels = channels,
            drop = drop
        ) for _ in range(depth)])

        self.projection = nn.Conv3d(
            in_channels = channels,
            out_channels = outChannels,
            kernel_size = 1
        )
    
    def forward(self, x):    # x: [B, T, C, H // 2, W // 2]
        z = self.upsample(x.permute(0, 2, 1, 3, 4))

        for block in self.blocks:
            z = z + block(z)
        
        out = self.projection(z).permute(0, 2, 1, 3, 4).contiguous()
        return out        