import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------------------------
# utils
# ----------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * rnd

class DepthwiseConv3d(nn.Module):
    def __init__(self, channels: int, k=(1,3,3), p=(0,1,1)):
        super().__init__()
        self.dw = nn.Conv3d(channels, channels, kernel_size=k, padding=p, groups=channels)
    def forward(self, x):
        return self.dw(x)

# ----------------------------
# Positional Encoding (same API)
# ----------------------------
class PositionalEncodingLite(nn.Module):
    def __init__(self, T, H, W, dim, num_heads, fourier_bands=6, max_len=512):
        super().__init__()
        self.dim = dim
        self.T, self.H, self.W = T, H, W

        # absolute learnable PE
        self.pe_abs = nn.Parameter(torch.zeros(1, T, dim, H, W))
        nn.init.trunc_normal_(self.pe_abs, std=0.02)

        # relative bias tables (kept)
        self.bt = nn.Parameter(torch.zeros(1, num_heads, max_len, max_len))
        self.bh = nn.Parameter(torch.zeros(1, num_heads, max_len, max_len))
        self.bw = nn.Parameter(torch.zeros(1, num_heads, max_len, max_len))

        # Fourier features (cached 1D encodings)
        self.bands = int(fourier_bands)
        freqs = (2.0 ** torch.arange(self.bands, dtype=torch.float32)) * math.pi
        self.register_buffer('freqs', freqs)  # [B]

        self.fourier_proj = nn.Linear(3 * 2 * self.bands, dim)
        self.fourier_scale = nn.Parameter(torch.tensor(0.2))  # learnable scale

        # precompute 1D encodings (sin/cos) for t/h/w (stored as buffers)
        t = torch.linspace(-1, 1, T)
        h = torch.linspace(-1, 1, H)
        w = torch.linspace(-1, 1, W)

        def enc_1d(coord):  # [L] -> [L, 2B]
            f = coord[:, None] * freqs[None, :]        # [L,B]
            return torch.cat([f.sin(), f.cos()], -1)   # [L,2B]

        self.register_buffer('enc_t_1d', enc_1d(t))  # [T,2B]
        self.register_buffer('enc_h_1d', enc_1d(h))  # [H,2B]
        self.register_buffer('enc_w_1d', enc_1d(w))  # [W,2B]

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        # safety: shapes must match initialized grid (kept behavior)
        assert T <= self.T and H <= self.H and W <= self.W, "Input grid exceeds initialized size"

        # add absolute PE
        x = x + self.pe_abs[:, :T, :C, :H, :W].to(dtype=x.dtype, device=x.device)

        # build concatenated 3D Fourier features from cached 1D (cheap broadcast)
        enc_t = self.enc_t_1d[:T].to(x.dtype, x.device)[:, None, None, :]  # [T,1,1,2B]
        enc_h = self.enc_h_1d[:H].to(x.dtype, x.device)[None, :, None, :]  # [1,H,1,2B]
        enc_w = self.enc_w_1d[:W].to(x.dtype, x.device)[None, None, :, :]  # [1,1,W,2B]
        enc = torch.cat([
            enc_t.expand(T, H, W, -1),
            enc_h.expand(T, H, W, -1),
            enc_w.expand(T, H, W, -1)
        ], dim=-1)  # [T,H,W,6B]

        feat = self.fourier_proj(enc).permute(3,0,1,2).unsqueeze(0)  # [1,C,T,H,W]
        feat = feat[:, :C].permute(0,2,1,3,4).contiguous()           # [1,T,C,H,W]
        x = x + self.fourier_scale * feat.expand(B, -1, -1, -1, -1)

        # relative bias slices, cast to attn dtype later
        bt = self.bt[..., :T, :T]
        bh = self.bh[..., :H, :H]
        bw = self.bw[..., :W, :W]
        return x, bt, bh, bw

# ----------------------------
# Axial Attention (same API)
# ----------------------------
class AxialAttention(nn.Module):
    def __init__(self, dim, num_heads, axis):
        super().__init__()
        self.dim = dim
        self.heads = num_heads
        self.axis = axis
        self.dh = dim // num_heads
        self.scale = self.dh ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj_drop = nn.Dropout(0.0)

        # chunk to control peak mem: number of axial "lines" per step
        self._line_chunk = 2048  # tune if needed

    def _reshape_axis(self, x):
        # x: [B,T,C,H,W] -> seq: [N, L, C], and inverse info
        B, T, C, H, W = x.shape
        if self.axis == 'time':
            N, L = B * H * W, T
            seq = x.permute(0, 3, 4, 1, 2).reshape(N, L, C)
            back = ('time', B, T, C, H, W)
        elif self.axis == 'height':
            N, L = B * T * W, H
            seq = x.permute(0, 1, 4, 3, 2).reshape(N, L, C)
            back = ('height', B, T, C, H, W)
        elif self.axis == 'width':
            N, L = B * T * H, W
            seq = x.permute(0, 1, 3, 4, 2).reshape(N, L, C)
            back = ('width', B, T, C, H, W)
        else:
            raise ValueError(f"Unknown axis: {self.axis}")
        return seq, back, N, L

    def _invert_axis(self, out, back):
        axis, B, T, C, H, W = back
        if axis == 'time':
            return out.view(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()
        if axis == 'height':
            return out.view(B, T, W, H, C).permute(0, 1, 4, 3, 2).contiguous()
        if axis == 'width':
            return out.view(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()

    def forward(self, x, rel_bias):   # x: [B,T,C,H,W], rel_bias: [1,Hd,L,L]
        seq, back, N, L = self._reshape_axis(x)
        Hd, Dh = self.heads, self.dh

        qkv = self.to_qkv(seq).chunk(3, dim=-1)  # [N,L,C] x3
        q, k, v = [t.view(N, L, Hd, Dh).transpose(1, 2).contiguous() for t in qkv]  # [N,Hd,L,Dh]

        # chunk over N to control memory
        outputs = []
        step = self._line_chunk
        rb = rel_bias.to(dtype=q.dtype, device=q.device)  # [1,Hd,L,L]
        for i in range(0, N, step):
            qi = q[i:i+step]        # [n,Hd,L,Dh]
            ki = k[i:i+step]
            vi = v[i:i+step]

            attn = torch.matmul(qi, ki.transpose(-2, -1)) * self.scale  # [n,Hd,L,L]
            attn = attn + rb  # broadcast over batch-lines
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out = torch.matmul(attn, vi)  # [n,Hd,L,Dh]
            out = out.transpose(1, 2).reshape(out.shape[0], L, Hd * Dh)  # [n,L,C]
            out = self.to_out(out)
            out = self.proj_drop(out)
            outputs.append(out)

        out = torch.cat(outputs, dim=0)  # [N,L,C]
        return self._invert_axis(out, back)

# ----------------------------
# ASTRA Block (same API)
# ----------------------------
class ASTRA_Block(nn.Module):
    def __init__(self, dim, T, H, W, heads=16, drop=0.0):
        super().__init__()
        self.posenc = PositionalEncodingLite(T, H, W, dim, heads)

        # pre-norm in channel-last domain
        self.norm_t = nn.LayerNorm(dim)
        self.norm_h = nn.LayerNorm(dim)
        self.norm_w = nn.LayerNorm(dim)

        self.attn_t = AxialAttention(dim, heads, 'time')
        self.attn_h = AxialAttention(dim, heads, 'height')
        self.attn_w = AxialAttention(dim, heads, 'width')

        # learnable residual mix (kept) + tiny scales for stability
        self.weight_t = nn.Parameter(torch.tensor(1.0))
        self.weight_h = nn.Parameter(torch.tensor(1.0))
        self.weight_w = nn.Parameter(torch.tensor(1.0))
        self.res_scale_attn = nn.Parameter(torch.tensor(0.1))
        self.res_scale_ffn  = nn.Parameter(torch.tensor(0.1))

        # MLP: 1x1 -> depthwise(1x3x3) -> 1x1, zero-init last
        self.ffn = nn.Sequential(
            nn.Conv3d(dim, dim * 4, 1),
            nn.GELU(),
            DepthwiseConv3d(dim * 4, k=(1,3,3), p=(0,1,1)),
            nn.GELU(),
            nn.Conv3d(dim * 4, dim, 1),
            nn.Dropout(drop)
        )
        # zero-init last conv for mild residual at start
        nn.init.zeros_(self.ffn[-2].weight)
        if self.ffn[-2].bias is not None:
            nn.init.zeros_(self.ffn[-2].bias)

        self.drop_path_attn = DropPath(drop)
        self.drop_path_ffn  = DropPath(drop)

    def forward(self, x):  # [B,T,C,H,W]
        x, bt, bh, bw = self.posenc(x)

        # channel-last for per-axis LayerNorm
        y = x.permute(0, 1, 3, 4, 2).contiguous()  # [B,T,H,W,C]

        t = self.attn_t(self.norm_t(y).permute(0, 1, 4, 2, 3), bt)  # [B,T,C,H,W]
        h = self.attn_h(self.norm_h(y).permute(0, 1, 4, 2, 3), bh)
        w = self.attn_w(self.norm_w(y).permute(0, 1, 4, 2, 3), bw)

        a = self.weight_t * t + self.weight_h * h + self.weight_w * w
        z = x + self.drop_path_attn(self.res_scale_attn * a)  # residual

        zc = z.permute(0, 2, 1, 3, 4).contiguous()  # [B,C,T,H,W]
        zc = zc + self.drop_path_ffn(self.res_scale_ffn * self.ffn(zc))
        z = zc.permute(0, 2, 1, 3, 4).contiguous()  # back to [B,T,C,H,W]
        return z
