import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from module.model.transformer.utils import DropPath, DepthwiseConv3d





class PositionalEncoding3D(nn.Module):
    def __init__(self,
                 T : int,
                 H : int,
                 W : int,
                 C : int,
                 heads : int,
                 bands : int = 6,
                 maxLen : int = 512
                 ):
        
        super().__init__()

        self.T = T
        self.H = H
        self.W = W
        self.C = C

        self.absolute = nn.Parameter(torch.zeros(1, T, C, H, W))
        nn.init.trunc_normal_(
            tensor = self.absolute,
            std = 0.02
        )

        self.biasT = nn.Parameter(torch.zeros(1, heads, maxLen, maxLen))
        self.biasW = nn.Parameter(torch.zeros(1, heads, maxLen, maxLen))
        self.biasH = nn.Parameter(torch.zeros(1, heads, maxLen, maxLen))

        frequencies = (2.0 ** torch.arange(bands)) * math.pi
        self.register_buffer('frequencies', frequencies)
        self.projection = nn.Linear(3 * 2 * bands, C)
        self.alpha = nn.Parameter(torch.tensor(0.2))


        def encode1D(L):
            c = torch.linspace(
                start = -1,
                end = 1,
                steps = L
            )[:, None] * frequencies[None, :]

            return torch.cat([c.sin(), c.cos()], dim = -1)    # [L, 2 * Bands]
        
        self.register_buffer('encodeT', encode1D(T))    # [T, 2 * Bands]
        self.register_buffer('encodeH', encode1D(H))    # [H, 2 * Bands]
        self.register_buffer('encodeW', encode1D(W))    # [W, 2 * Bands]

    def forward(self, x):    # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        assert T <= self.T and H <= self.H and W <= self.W

        x = x + self.absolute[:, :T, :C, :H, :W].to(device = x.device, dtype = x.dtype)

        encodeT = self.encodeT[:T].to(device = x.device, dtype = x.dtype)[:, None, None, :]    # [T, 1, 1, 2 * Bands]
        encodeH = self.encodeH[:H].to(device = x.device, dtype = x.dtype)[None, :, None, :]    # [1, H, 1, 2 * Bands]
        encodeW = self.encodeW[:W].to(device = x.device, dtype = x.dtype)[None, None, :, :]    # [1, 1, W, 2 * Bands]

        encodeTotal = torch.cat([
            encodeT.expand(T, H, W, -1),
            encodeH.expand(T, H, W, -1),
            encodeW.expand(T, H, W, -1)
        ], dim = -1)    # encodeTotal: [T, H, W, 6 * Bands]


        feature = self.projection(encodeTotal)    # feature: [T, H, W, C]
        feature = feature.permute(0, 3, 1, 2).unsqueeze(0)    # feature: [1, T, C, H, W]
        feature = feature[:, :, :C].contiguous()

        x = x + self.alpha * feature.expand(B, -1, -1, -1, -1)

        biasT = self.biasT[..., :T, :T]
        biasH = self.biasH[..., :H, :H]
        biasW = self.biasW[..., :W, :W]


        return x, biasT, biasH, biasW


class AxialAttention(nn.Module):
    def __init__(self,
                 C : int,
                 heads : int,
                 axis : str,
                 lineChunk : int = 2048
                 ):
        
        super().__init__()

        self.C = C
        self.heads = heads
        self.axis = axis

        self.dim = C // heads
        self.scale = self.dim ** -0.5

        self.qkv = nn.Linear(
            in_features = C,
            out_features = 3 * C,
            bias = True
        )

        self.projection = nn.Linear(
            in_features = C,
            out_features = C,
            bias = True
        )

        self.lineChunk = lineChunk
    
    def _reshape(self, x):    # x: [B, T, C, H, W] -> [N, L, C]
        B, T, C, H, W = x.shape

        if self.axis == "time":
            sequence = x.permute(0, 3, 4, 1, 2)
            sequence = sequence.reshape(B * H * W, T, C)
            info = ("time", B, T, C, H, W)
        elif self.axis == "height":
            sequence = x.permute(0, 1, 4, 3, 2)
            sequence = sequence.reshape(B * T * W, H, C)
            info = ("height", B, T, C, H, W)
        elif self.axis == "width":
            sequence = x.permute(0, 1, 3, 4, 2)
            sequence = sequence.reshape(B * T * H, W, C)
            info = ("width", B, T, C, H, W)
        else:
            raise RuntimeError(f"Axis {self.axis} does not exist")

        return sequence, info
    
    def _invert(self, y, info):    # y: [N, L, C]
        axis, B, T, C, H, W = info

        if axis == "time":
            return y.view(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()
        elif axis == "height":
            return y.view(B, T, W, H, C).permute(0, 1, 4, 3, 2).contiguous()
        elif axis == "width":
            return y.view(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()
        else:
            raise RuntimeError(f"Axis {axis} does not exist")
    
    def forward(self, x, relativeBias):    # x: [B, T, C, H, W]
        sequence, info = self._reshape(x)    # sequence: [N, L, C]
        N, L, C = sequence.shape

        qkv = self.qkv(sequence)
        q, k, v = torch.chunk(
            input = qkv,
            chunks = 3,
            dim = -1
        )    # q, k, v: [N, L, C]

        def split(t):
            return t.view(N, L, self.heads, self.dim).transpose(1, 2).contiguous()
        
        q, k, v = map(split, (q, k, v))    # q, k, v: [N, h, L, d]


        outChunks = []

        relativeBias = relativeBias.to(device = q.device, dtype = q.dtype)    # relativeBias: [1, h, L, L]
        step = self.lineChunk
        for i in range(0, N, step):
            qi, ki, vi = q[i:i+step], k[i:i+step], v[i:i+step]
            attn = (qi @ ki.transpose(-2, -1)) * self.scale    # attn: [n h, L, L]
            attn = (attn + relativeBias).softmax(dim = -1)

            oi = attn @ vi    # oi: [n, h, L, d]
            oi = oi.transpose(1, 2).reshape(oi.size(0), L, C)    # oi: [n, L, C]
            outChunks.append(self.projection(oi))    # self.projection: [n, L, C]
        
        out = torch.cat(outChunks, dim = 0)    # out: [N, L, C]

        return self._invert(
            y = out,
            info = info
        )    # self._invert: [B, T, C, H, W]
    

class MultiLayerPerceptron3D(nn.Module):
    def __init__(self,
                 C : int,
                 dropout : float = 0.0
                 ):
        
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(
                in_channels = C,
                out_channels = 4 * C,
                kernel_size = 1
            ),
            nn.GELU(),

            DepthwiseConv3d(
                channels = 4 * C,
                kernel = (1, 3, 3),
                padding = (0, 1, 1)
            ),
            nn.GELU(),

            nn.Conv3d(
                in_channels = 4 * C,
                out_channels = C,
                kernel_size = 1
            ),
            nn.Dropout(dropout)
        )
        nn.init.zeros_(self.net[-2].weight)

        if self.net[-2].bias is not None:
            nn.init.zeros_(self.net[-2].bias)
    
    def forward(self, x):
        return self.net(x)


class AxialBlockMLP(nn.Module):
    def __init__(self,
                 C : int,
                 T : int,
                 H : int,
                 W : int,
                 heads : int = 16,
                 drop : float = 0.0
                 ):

        super().__init__()

        self.positionalEncoding = PositionalEncoding3D(
            T = T,
            H = H,
            W = W,
            C = C,
            heads = heads
        )

        self.preMLP = MultiLayerPerceptron3D(C = C, dropout = drop)
        self.betaPre = nn.Parameter(torch.tensor(0.1))
        self.dropPre = DropPath(drop)

        self.normalizeT = nn.LayerNorm(C)
        self.normalizeH = nn.LayerNorm(C)
        self.normalizeW = nn.LayerNorm(C)

        self.attnT = AxialAttention(
            C = C,
            heads = heads,
            axis = "time"
        )
        self.attnH = AxialAttention(
            C = C,
            heads = heads,
            axis = "height"
        )
        self.attnW = AxialAttention(
            C = C,
            heads = heads,
            axis = "width"
        )

        self.weightT = nn.Parameter(torch.tensor(1.0))
        self.weightH = nn.Parameter(torch.tensor(1.0))
        self.weightW = nn.Parameter(torch.tensor(1.0))

        self.betaAttn = nn.Parameter(torch.tensor(0.1))
        self.betaFnn1 = nn.Parameter(torch.tensor(0.1))
        self.betaFnn2 = nn.Parameter(torch.tensor(0.1))

        self.dropAttn = DropPath(drop)
        self.dropFnn1 = DropPath(drop)
        self.dropFnn2 = DropPath(drop)

        self.mlp1 = MultiLayerPerceptron3D(C = C, dropout = drop)
        self.mlp2 = MultiLayerPerceptron3D(C = C, dropout = drop)

        self.postAttnConv = nn.Conv3d(
            in_channels = C,
            out_channels = C,
            kernel_size = 1,
            bias = True
        )
        nn.init.trunc_normal_(self.postAttnConv.weight, std = 0.02)
        if self.postAttnConv.bias is not None:
            nn.init.zeros_(self.postAttnConv.bias)
    
    @staticmethod
    def _layernorm_last(x, layerNorm):
        xCl = x.permute(0, 1, 3, 4, 2).contiguous()
        xCl = layerNorm(xCl)
        return xCl.permute(0, 1, 4, 2, 3).contiguous()

    def forward(self, x):    # x: [B, T, C, H, W]
        x, biasT, biasH, biasW = self.positionalEncoding(x)

        xPre = x.permute(0, 2, 1, 3, 4).contiguous()
        xPre = xPre + self.dropPre(self.betaPre * self.preMLP(xPre))
        x = xPre.permute(0, 2, 1, 3, 4).contiguous()

        xT = self._layernorm_last(x, self.normalizeT)
        xH = self._layernorm_last(x, self.normalizeH)
        xW = self._layernorm_last(x, self.normalizeW)

        t = self.attnT(xT, biasT)
        h = self.attnH(xH, biasH)
        w = self.attnW(xW, biasW)

        a = (self.weightT * t) + (self.weightH * h) + (self.weightW * w)    # a: [B, T, C, H, W]

        aConv = a.permute(0, 2, 1, 3, 4).contiguous()    # aConv: [B, C, T, H, W]
        z1Conv = self.dropAttn(self.postAttnConv(aConv) * self.betaAttn)

        z2 = z1Conv + self.dropFnn1(self.betaFnn1 * self.mlp1(z1Conv))
        z3 = z2     + self.dropFnn2(self.betaFnn2 * self.mlp2(z2))


        out = z3.permute(0, 2, 1, 3, 4).contiguous()    # out: [B, T, C, H, W]
        return out + x
