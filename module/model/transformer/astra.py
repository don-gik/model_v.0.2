import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncodingLite(nn.Module):
    def __init__(self, T, H, W, dim, num_heads, fourier_bands=6, max_len=512):
        super().__init__()
        self.dim = dim
        self.T, self.H, self.W = T, H, W
        self.pe_abs = nn.Parameter(torch.zeros(1, T, dim, H, W))  # dim 위치 변경
        nn.init.trunc_normal_(self.pe_abs, std=0.02)

        self.bt = nn.Parameter(torch.zeros(1, num_heads, max_len, max_len))
        self.bh = nn.Parameter(torch.zeros(1, num_heads, max_len, max_len))
        self.bw = nn.Parameter(torch.zeros(1, num_heads, max_len, max_len))

        self.bands = fourier_bands
        freqs = 2 ** torch.arange(fourier_bands, dtype=torch.float32) * math.pi
        self.register_buffer('freqs', freqs)
        self.fourier_proj = nn.Linear(3 * 2 * fourier_bands, dim)

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x + self.pe_abs.expand(B, -1, -1, -1, -1)  # ✅ 바로 더함

        # Create 3D grid
        t = torch.linspace(-1, 1, T, device=x.device)
        h = torch.linspace(-1, 1, H, device=x.device)
        w = torch.linspace(-1, 1, W, device=x.device)
        tt, hh, ww = torch.meshgrid(t, h, w, indexing='ij')

        def encode(coord):
            f = coord[..., None] * self.freqs  # [T,H,W,B]
            return torch.cat([f.sin(), f.cos()], dim=-1)  # [T,H,W,2B]

        enc_t = encode(tt)
        enc_h = encode(hh)
        enc_w = encode(ww)
        enc = torch.cat([enc_t, enc_h, enc_w], dim=-1)  # [T,H,W,6B]
        feat = self.fourier_proj(enc).permute(3, 0, 1, 2).unsqueeze(0)  # [1,C,T,H,W]
        feat = feat.permute(0, 2, 1, 3, 4)  # → [1,T,C,H,W]
        x = x + 0.2 * feat.expand(B, -1, -1, -1, -1)

        return x, self.bt[..., :T, :T], self.bh[..., :H, :H], self.bw[..., :W, :W]


class AxialAttention(nn.Module):
    def __init__(self, dim, num_heads, axis):
        super().__init__()
        self.dim = dim
        self.heads = num_heads
        self.axis = axis
        self.scale = (dim // num_heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, rel_bias):
        B, T, C, H, W = x.shape
        if self.axis == 'time':
            N, L = B * H * W, T
            seq = x.permute(0, 3, 4, 1, 2).reshape(N, L, C)
        elif self.axis == 'height':
            N, L = B * T * W, H
            seq = x.permute(0, 1, 4, 3, 2).reshape(N, L, C)
        elif self.axis == 'width':
            N, L = B * T * H, W
            seq = x.permute(0, 1, 3, 4, 2).reshape(N, L, C)
        else:
            raise ValueError(f"Unknown axis: {self.axis}")

        qkv = self.to_qkv(seq).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(N, L, self.heads, C // self.heads).transpose(1, 2), qkv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + rel_bias
        attn = attn.softmax(dim=-1)
        out = self.to_out((attn @ v).transpose(1, 2).reshape(N, L, C))

        if self.axis == 'time':
            return out.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)
        elif self.axis == 'height':
            return out.view(B, T, W, H, C).permute(0, 1, 4, 3, 2)
        elif self.axis == 'width':
            return out.view(B, T, H, W, C).permute(0, 1, 4, 2, 3)

class ASTRA_Block(nn.Module):
    def __init__(self, dim, T, H, W, heads=16, drop=0.0):
        super().__init__()
        self.posenc = PositionalEncodingLite(T, H, W, dim, heads)

        self.norm_t = nn.LayerNorm(dim)
        self.norm_h = nn.LayerNorm(dim)
        self.norm_w = nn.LayerNorm(dim)

        self.attn_t = AxialAttention(dim, heads, 'time')
        self.attn_h = AxialAttention(dim, heads, 'height')
        self.attn_w = AxialAttention(dim, heads, 'width')

        self.weight_t = nn.Parameter(torch.tensor(1.0))
        self.weight_h = nn.Parameter(torch.tensor(1.0))
        self.weight_w = nn.Parameter(torch.tensor(1.0))

        self.ffn = nn.Sequential(
            nn.Conv3d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv3d(dim * 4, dim, 1),
            nn.Dropout(drop)
        )

    def forward(self, x):  # [B,T,C,H,W]
        x, bt, bh, bw = self.posenc(x)

        y = x.permute(0, 1, 3, 4, 2)  # [B,T,H,W,C]

        t = self.attn_t(self.norm_t(y).permute(0, 1, 4, 2, 3), bt)
        h = self.attn_h(self.norm_h(y).permute(0, 1, 4, 2, 3), bh)
        w = self.attn_w(self.norm_w(y).permute(0, 1, 4, 2, 3), bw)

        a = self.weight_t * t + self.weight_h * h + self.weight_w * w
        z = x + 0.1 * a  # residual + scaling

        z = z.permute(0, 2, 1, 3, 4)             # ➜ [B,C,T,H,W]
        z = z + 0.1 * self.ffn(z)                # ➜ FFN 처리
        z = z.permute(0, 2, 1, 3, 4)             # ➜ [B,T,C,H,W]로 복구
        return z
