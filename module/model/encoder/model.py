import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

# class Encoder(nn.Module):
#     def __init__(self, in_channels=4, embed_dim=64, mid_dim=16, use_coords=False):
#         super().__init__()
#         self.use_coords = use_coords
#         self.embed_dim = embed_dim
#         in_ch = in_channels + (2 if use_coords else 0)
#         # Use augmented channel count when coords are enabled
#         self.temporal = nn.Conv2d(in_ch, mid_dim, kernel_size=3, padding=1)
#         self.spatial  = nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=2, padding=1)
#         self.dilated  = nn.Sequential(
#             nn.Conv2d(mid_dim, mid_dim, 3, padding=2, dilation=2, groups=mid_dim),
#             nn.BatchNorm2d(mid_dim), nn.GELU(),
#             nn.Conv2d(mid_dim, mid_dim, 1)
#         )
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(mid_dim, mid_dim*2, 1), nn.GELU(), nn.Conv2d(mid_dim*2, mid_dim, 1)
#         )
#         self.to_embed = nn.Conv2d(mid_dim, embed_dim, 1)
#         self.norm = nn.LayerNorm(embed_dim)

#     def add_coord_channels(self, x):
#         B, C, H, W = x.shape
#         dev = x.device
#         i = torch.linspace(-1, 1, H, device=dev).view(1,1,H,1).expand(B,1,H,W)
#         j = torch.linspace(-1, 1, W, device=dev).view(1,1,1,W).expand(B,1,H,W)
#         return torch.cat([x, i, j], dim=1)

#     def forward(self, x):
#         x = x.squeeze(1)                           # [B, C, H, W]
#         if self.use_coords: x = self.add_coord_channels(x)  # 입력 채널만 +2
#         x = self.temporal(x); x = self.spatial(x)
#         x = self.dilated(x) + x
#         x = self.bottleneck(x) + x
#         x = self.to_embed(x)                       # [B, 4, H/2, W/2]
#         B, D, H, W = x.shape
#         x = x.permute(0,2,3,1).reshape(B*H*W, D)
#         x = self.norm(x).reshape(B,H,W,D).permute(0,3,1,2)
#         return x

#     @property
#     def out_dim(self):
#         return self.embed_dim


# --- utils for better encoder using MHSA ---
def window_partition(x, windowHeight : int, windowWidth : int):
    # x: [B, D, H, W] -> [B * windowCount, N, D]

    B, D, H, W = x.shape
    assert H % windowHeight == 0 and W % windowWidth == 0

    x = x.view(B, D, H // windowHeight, windowHeight, W // windowWidth, windowWidth)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()

    BatchNWindows = B * (H // windowHeight) * (W // windowWidth)
    N = windowWidth * windowHeight

    return x.view(BatchNWindows, N, D), (B, D, H, W)    # x: [BnW, N, D]

def window_reverse(xWindow, windowHeight : int, windowWidth : int, shape):
    # xWindow: [BnW, N, D] -> [B, D, H, W]

    B, D, H, W = shape
    Hpartition, Wpartition = H // windowHeight, W // windowWidth
    xWindow = xWindow.view(B, Hpartition, Wpartition, windowHeight, windowWidth, D)
    xWindow = xWindow.permute(0, 5, 1, 3, 2, 4).contiguous()

    return xWindow.view(B, D, H, W)


class MHSAAttn(nn.Module):
    def __init__(self, dim : int, heads : int):
        super().__init__()
        
        self.dim = dim
        self.heads = heads
        self.dimByHeads = dim // heads

        self.qkv = nn.Linear(dim, dim * 3, bias = False)
        self.proj = nn.Linear(dim, dim, bias = False)

        self.scale = self.dh ** -0.5
    
    def forward(self, x):    # x: [B * nw, N, D]
        BatchNWindow, N, D = x.shape

        qkv = self.qkv(x)    # qkv: [B * nw, N, 3 * D]
        qkv = qkv.view(BatchNWindow, N, 3, self.heads, self.dimByHeads)    # qkv: [B * nw, N, 3, h, D // h]
        qkv = qkv.transpose(1, 3)    # qkv: [B * nw, h, 3, N, D // h]

        q, k, v = qkv.unbind(dim = 2)    # q, k, v: [B * nw, h, N, D // h]

        attn = (q @ k.transpose(2, 3)) * self.scale    # attn: [B * nw, h, N, N]
        attn = attn.softmax(dim = -1)    # attn: [B * nw, h, N, N]

        out = attn @ v    # out: [B * nw, h, N, dh]
        out = out.transpose(1, 2).contiguous()    # out: [B * nw, N, h, dh]
        out = out.view(BatchNWindow, N, D)    # out: [B * nw, N, D]

        return self.proj(out)   # out: [B * nw, N, D]


class WindowBlock(nn.Module):
    def __init__(self, 
                 dim : int, 
                 heads : int, 
                 windowHeight : int,
                 windowWidth : int,
                 mlpRatio : float = 2.0,
                 dropout : float = 0.1
                 ):
        
        self.windowHeight = windowHeight
        self.windowWidth = windowWidth

        self.layerNorm1 = nn.LayerNorm(dim)
        self.layerNorm2 = nn.LayerNorm(dim)

        self.attn = MHSAAttn(dim = dim, heads = heads)

        hiddenSize = int(dim * mlpRatio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hiddenSize),
            nn.GELU(),
            nn.Linear(hiddenSize, dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):    # x: [B, D, H, W]
        xWindow, shape = window_partition(x = x,
                                          windowHeight = self.windowHeight,
                                          windowWidth = self.windowWidth)    # xWindow: [B * nw, N, D]
        
        heads = xWindow + self.dropout(self.attn(self.layerNorm1(xWindow)))    # heads: [B * nw, N, D]
        heads = heads + self.dropout(self.mlp(self.layerNorm2(heads)))    # heads: [B * nw, N, D]

        return window_reverse(xWindow = heads,
                              windowHeight = self.windowHeight,
                              windowWidth = self.windowWidth,
                              shape = shape)    # return: [B, D, H, W]
    

class Encoder(nn.Module):
    def __init__(self,
                 inChannels : int = 4,
                 embedDim : int = 64,
                 midDim : int = 16,
                 heads : int = 4,
                 depth : int = 2,
                 window : Tuple[int, int] = (5, 10),
                 mlpRatio = 2.0,
                 dropout = 0.1):
        
        super().__init__()

        self.embedDim = embedDim

        self.temporal = nn.Conv2d(in_channels = inChannels,
                                  out_channels = midDim,
                                  kernel_size = 3,
                                  padding = 1)
        self.spatial = nn.Conv2d(in_channels = inChannels,
                                 out_channels = midDim,
                                 kernel_size = 3,
                                 stride = 2,
                                 padding = 1)
        self.dilated = nn.Sequential(
            nn.Conv2d(in_channels = midDim,
                      out_channels = midDim,
                      kernel_size = 3,
                      padding = 1,
                      dilation = 2,
                      groups = midDim),
            nn.BatchNorm2D(midDim),
            nn.GELU(),
            nn.Conv2d(midDim, midDim, 1)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels = midDim,
                      out_channels = midDim * 2,
                      kernel_size = 1),
            nn.GELU(),
            nn.Conv2d(in_channels = midDim * 2,
                      out_channels = midDim,
                      kernel_size = 1)
        )

        self.toEmbed = nn.Conv2d(in_channels = midDim,
                                 out_channels = embedDim,
                                 kernel_size = 1)
        

        windowWidth, windowHeight = window
        
        blocksStacked = [WindowBlock(dim = embedDim,
                                     heads = heads,
                                     windowHeight = windowHeight,
                                     windowWidth = windowWidth,
                                     mlpRatio = mlpRatio,
                                     dropout = dropout) 
                                     for _ in range(depth)]
        self.blocks = nn.ModuleList(blocksStacked)


        self.finalLayerNorm = nn.LayerNorm(embedDim)
    

    def forward(self, x):   # [B, 1, C, H, W]
        x = x.squeeze(1)    # [B, C, H, W]

        x = self.temporal(x)
        x = self.spatial(x)
        x = self.dilated(x) + x
        x = self.bottleneck(x) + x
        x = self.toEmbed(x)    # [B, C, H // 2, W // 2]

        for block in self.blocks:    # windowed MSHA + MLP
            x = block(x)


        x = x.permute(0, 2, 3, 1)   # [B, H // 2, W // 2, C]
        x = self.finalLayerNorm(x)
        x = x.permute(0, 3, 1, 2)   # [B, C, H // 2, W // 2]
        return x
    
    
    @property
    def out_dim(self):
        return self.embedDim