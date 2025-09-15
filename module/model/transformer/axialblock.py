# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# # ----------------------------
# # utils
# # ----------------------------
# class DropPath(nn.Module):
#     def __init__(self, drop_prob: float = 0.0):
#         super().__init__()
#         self.drop_prob = float(drop_prob)

#     def forward(self, x):
#         if self.drop_prob == 0.0 or not self.training:
#             return x
#         keep = 1 - self.drop_prob
#         shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#         rnd = x.new_empty(shape).bernoulli_(keep).div_(keep)
#         return x * rnd

# class DepthwiseConv3d(nn.Module):
#     def __init__(self, channels: int, k=(1,3,3), p=(0,1,1)):
#         super().__init__()
#         self.dw = nn.Conv3d(channels, channels, kernel_size=k, padding=p, groups=channels)
#     def forward(self, x):
#         return self.dw(x)

# # ----------------------------
# # Positional Encoding (same API)
# # ----------------------------
# class PositionalEncodingLite(nn.Module):
#     def __init__(self, T, H, W, dim, num_heads, fourier_bands=6, max_len=512):
#         super().__init__()
#         self.dim = dim
#         self.T, self.H, self.W = T, H, W

#         # absolute learnable PE
#         self.pe_abs = nn.Parameter(torch.zeros(1, T, dim, H, W))
#         nn.init.trunc_normal_(self.pe_abs, std=0.02)

#         # relative bias tables (kept)
#         self.bt = nn.Parameter(torch.zeros(1, num_heads, max_len, max_len))
#         self.bh = nn.Parameter(torch.zeros(1, num_heads, max_len, max_len))
#         self.bw = nn.Parameter(torch.zeros(1, num_heads, max_len, max_len))

#         # Fourier features (cached 1D encodings)
#         self.bands = int(fourier_bands)
#         freqs = (2.0 ** torch.arange(self.bands, dtype=torch.float32)) * math.pi
#         self.register_buffer('freqs', freqs)  # [B]

#         self.fourier_proj = nn.Linear(3 * 2 * self.bands, dim)
#         self.fourier_scale = nn.Parameter(torch.tensor(0.2))  # learnable scale

#         # precompute 1D encodings (sin/cos) for t/h/w (stored as buffers)
#         t = torch.linspace(-1, 1, T)
#         h = torch.linspace(-1, 1, H)
#         w = torch.linspace(-1, 1, W)

#         def enc_1d(coord):  # [L] -> [L, 2B]
#             f = coord[:, None] * freqs[None, :]        # [L,B]
#             return torch.cat([f.sin(), f.cos()], -1)   # [L,2B]

#         self.register_buffer('enc_t_1d', enc_1d(t))  # [T,2B]
#         self.register_buffer('enc_h_1d', enc_1d(h))  # [H,2B]
#         self.register_buffer('enc_w_1d', enc_1d(w))  # [W,2B]

#     def forward(self, x):  # x: [B, T, C, H, W]
#         B, T, C, H, W = x.shape
#         # safety: shapes must match initialized grid (kept behavior)
#         assert T <= self.T and H <= self.H and W <= self.W, "Input grid exceeds initialized size"

#         # add absolute PE
#         x = x + self.pe_abs[:, :T, :C, :H, :W].to(dtype=x.dtype, device=x.device)

#         # build concatenated 3D Fourier features from cached 1D (cheap broadcast)
#         enc_t = self.enc_t_1d[:T].to(x.dtype, x.device)[:, None, None, :]  # [T,1,1,2B]
#         enc_h = self.enc_h_1d[:H].to(x.dtype, x.device)[None, :, None, :]  # [1,H,1,2B]
#         enc_w = self.enc_w_1d[:W].to(x.dtype, x.device)[None, None, :, :]  # [1,1,W,2B]
#         enc = torch.cat([
#             enc_t.expand(T, H, W, -1),
#             enc_h.expand(T, H, W, -1),
#             enc_w.expand(T, H, W, -1)
#         ], dim=-1)  # [T,H,W,6B]

#         feat = self.fourier_proj(enc).permute(3,0,1,2).unsqueeze(0)  # [1,C,T,H,W]
#         feat = feat[:, :C].permute(0,2,1,3,4).contiguous()           # [1,T,C,H,W]
#         x = x + self.fourier_scale * feat.expand(B, -1, -1, -1, -1)

#         # relative bias slices, cast to attn dtype later
#         bt = self.bt[..., :T, :T]
#         bh = self.bh[..., :H, :H]
#         bw = self.bw[..., :W, :W]
#         return x, bt, bh, bw

# # ----------------------------
# # Axial Attention (same API)
# # ----------------------------
# class AxialAttention(nn.Module):
#     def __init__(self, dim, num_heads, axis):
#         super().__init__()
#         self.dim = dim
#         self.heads = num_heads
#         self.axis = axis
#         self.dh = dim // num_heads
#         self.scale = self.dh ** -0.5
#         self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
#         self.to_out = nn.Linear(dim, dim, bias=True)
#         self.attn_drop = nn.Dropout(0.0)
#         self.proj_drop = nn.Dropout(0.0)

#         # chunk to control peak mem: number of axial "lines" per step
#         self._line_chunk = 2048  # tune if needed

#     def _reshape_axis(self, x):
#         # x: [B,T,C,H,W] -> seq: [N, L, C], and inverse info
#         B, T, C, H, W = x.shape
#         if self.axis == 'time':
#             N, L = B * H * W, T
#             seq = x.permute(0, 3, 4, 1, 2).reshape(N, L, C)
#             back = ('time', B, T, C, H, W)
#         elif self.axis == 'height':
#             N, L = B * T * W, H
#             seq = x.permute(0, 1, 4, 3, 2).reshape(N, L, C)
#             back = ('height', B, T, C, H, W)
#         elif self.axis == 'width':
#             N, L = B * T * H, W
#             seq = x.permute(0, 1, 3, 4, 2).reshape(N, L, C)
#             back = ('width', B, T, C, H, W)
#         else:
#             raise ValueError(f"Unknown axis: {self.axis}")
#         return seq, back, N, L

#     def _invert_axis(self, out, back):
#         axis, B, T, C, H, W = back
#         if axis == 'time':
#             return out.view(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()
#         if axis == 'height':
#             return out.view(B, T, W, H, C).permute(0, 1, 4, 3, 2).contiguous()
#         if axis == 'width':
#             return out.view(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()

#     def forward(self, x, rel_bias):   # x: [B,T,C,H,W], rel_bias: [1,Hd,L,L]
#         seq, back, N, L = self._reshape_axis(x)
#         Hd, Dh = self.heads, self.dh

#         qkv = self.to_qkv(seq).chunk(3, dim=-1)  # [N,L,C] x3
#         q, k, v = [t.view(N, L, Hd, Dh).transpose(1, 2).contiguous() for t in qkv]  # [N,Hd,L,Dh]

#         # chunk over N to control memory
#         outputs = []
#         step = self._line_chunk
#         rb = rel_bias.to(dtype=q.dtype, device=q.device)  # [1,Hd,L,L]
#         for i in range(0, N, step):
#             qi = q[i:i+step]        # [n,Hd,L,Dh]
#             ki = k[i:i+step]
#             vi = v[i:i+step]

#             attn = torch.matmul(qi, ki.transpose(-2, -1)) * self.scale  # [n,Hd,L,L]
#             attn = attn + rb  # broadcast over batch-lines
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)

#             out = torch.matmul(attn, vi)  # [n,Hd,L,Dh]
#             out = out.transpose(1, 2).reshape(out.shape[0], L, Hd * Dh)  # [n,L,C]
#             out = self.to_out(out)
#             out = self.proj_drop(out)
#             outputs.append(out)

#         out = torch.cat(outputs, dim=0)  # [N,L,C]
#         return self._invert_axis(out, back)

# # ----------------------------
# # ASTRA Block (same API)
# # ----------------------------
# class ASTRA_Block(nn.Module):
#     def __init__(self, dim, T, H, W, heads=16, drop=0.0):
#         super().__init__()
#         self.posenc = PositionalEncodingLite(T, H, W, dim, heads)

#         # pre-norm in channel-last domain
#         self.norm_t = nn.LayerNorm(dim)
#         self.norm_h = nn.LayerNorm(dim)
#         self.norm_w = nn.LayerNorm(dim)

#         self.attn_t = AxialAttention(dim, heads, 'time')
#         self.attn_h = AxialAttention(dim, heads, 'height')
#         self.attn_w = AxialAttention(dim, heads, 'width')

#         # learnable residual mix (kept) + tiny scales for stability
#         self.weight_t = nn.Parameter(torch.tensor(1.0))
#         self.weight_h = nn.Parameter(torch.tensor(1.0))
#         self.weight_w = nn.Parameter(torch.tensor(1.0))
#         self.res_scale_attn = nn.Parameter(torch.tensor(0.1))
#         self.res_scale_ffn  = nn.Parameter(torch.tensor(0.1))

#         # MLP: 1x1 -> depthwise(1x3x3) -> 1x1, zero-init last
#         self.ffn = nn.Sequential(
#             nn.Conv3d(dim, dim * 4, 1),
#             nn.GELU(),
#             DepthwiseConv3d(dim * 4, k=(1,3,3), p=(0,1,1)),
#             nn.GELU(),
#             nn.Conv3d(dim * 4, dim, 1),
#             nn.Dropout(drop)
#         )
#         # zero-init last conv for mild residual at start
#         nn.init.zeros_(self.ffn[-2].weight)
#         if self.ffn[-2].bias is not None:
#             nn.init.zeros_(self.ffn[-2].bias)

#         self.drop_path_attn = DropPath(drop)
#         self.drop_path_ffn  = DropPath(drop)

#     def forward(self, x):  # [B,T,C,H,W]
#         x, bt, bh, bw = self.posenc(x)

#         # channel-last for per-axis LayerNorm
#         y = x.permute(0, 1, 3, 4, 2).contiguous()  # [B,T,H,W,C]

#         t = self.attn_t(self.norm_t(y).permute(0, 1, 4, 2, 3), bt)  # [B,T,C,H,W]
#         h = self.attn_h(self.norm_h(y).permute(0, 1, 4, 2, 3), bh)
#         w = self.attn_w(self.norm_w(y).permute(0, 1, 4, 2, 3), bw)

#         a = self.weight_t * t + self.weight_h * h + self.weight_w * w
#         z = x + self.drop_path_attn(self.res_scale_attn * a)  # residual

#         zc = z.permute(0, 2, 1, 3, 4).contiguous()  # [B,C,T,H,W]
#         zc = zc + self.drop_path_ffn(self.res_scale_ffn * self.ffn(zc))
#         z = zc.permute(0, 2, 1, 3, 4).contiguous()  # back to [B,T,C,H,W]
#         return z



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

        x = x + self.absolute[:, :T, :C, :H, :W].to(x.dtype, x.device)

        encodeT = self.encodeT[:T].to(x.dtype, x.device)[:, None, None, :]    # [T, 1, 1, 2 * Bands]
        encodeH = self.encodeH[:H].to(x.dtype, x.device)[None, :, None, :]    # [1, H, 1, 2 * Bands]
        encodeW = self.encodeW[:W].to(x.dtype, x.device)[None, None, :, :]    # [1, 1, W, 2 * Bands]

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
    
    def _invert(self, y, info):    # y: [N, L, C]
        axis, B, T, C, H, W = info

        if axis == "time":
            return y.view(B, T, C, H, W).contiguous()
        elif axis == "height":
            return y.view(B, T, C, H, W).contiguous()
        elif axis == "width":
            return y.view(B, T, C, H, W).contiguous()
        else:
            raise RuntimeError(f"Axis {axis} does not exist")
    
    def forward(self, x, relativeBias):    # x: [B, T, C, H, W]
        sequence, info = self._reshape(x)    # sequence: [N, L, C]
        N, L, C = sequence.shape

        q, k, v = self.qkv(sequence).chunk(dim = 1)    # q, k, v: [N, L, C]

        def split(t):
            return t.view(N, L, self.h, self.d).transpose(1, 2).contiguous()
        
        q, k, v = map(split, (q, k, v))    # q, k, v: [N, h, L, d]


        outChunks = []

        relativeBias = relativeBias.to(dtype = q.dtype, device = q.device)    # relativeBias: [1, h, L, L]
        step = self.lineChunk
        for i in range(0, N, step):
            qi, ki, vi = q[i:i+step], k[i:i+step], v[i:i+step]
            attn = (qi @ ki.transpose(-2, -1)) * self.scale    # attn: [n h, L, L]
            attn = (attn + relativeBias).softmax(dim = 1)

            oi = attn @ vi    # oi: [n, h, L, d]
            oi = oi.transpose(1, 2).reshape(oi.size(0), L, C)    # oi: [n, L, C]
            outChunks.append(self.proj(oi))    # self.proj: [n, L, C]
        
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
                in_channels = 4 * C,
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
            axis = "time"
        )
        self.attnT = AxialAttention(
            C = C,
            heads = heads,
            axis = "time"
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
    

    def forward(self, x):    # x: [B, T, C, H, W]
        x, biasT, biasH, biasW = self.positionalEncoding(x)

        t = self.attnT(self.normalizeT(x), biasT)
        h = self.attnH(self.normalizeH(x), biasH)
        w = self.attnW(self.normalizeW(x), biasW)

        a = (self.weightT * t) + (self.weightH * h) + (self.weightW * w)    # a: [B, T, C, H, W]

        aConv = a.permute(0, 2, 1, 3, 4).contiguous()    # aConv: [B, C, T, H, W]
        z1Conv = self.postAttnConv(aConv)
        
        z2 = z1Conv + self.dropFnn1(self.betaFnn1 * self.mlp1(z1Conv))
        z3 = z2     + self.dropFnn2(self.betaFnn2 * self.mlp2(z2))


        out = z3.permute(0, 2, 1, 3, 4).contiguous()    # out: [B, T, C, H, W]
        return out
