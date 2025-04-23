# src/models/birdjepa.py
# minimalist BirdJEPA encoder with 8×64 patch grid and plain 1-D rotary PE
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass, field
from .rope import rope

print(">> birdjepa loaded from", __file__)

# ──────────────────────────────────────
#  config
# ──────────────────────────────────────
@dataclass
class BJConfig:
    # input
    n_mels: int = 128

    # four (freq, time) strided conv layers → F:128→8,  T:1000→64
    conv_ch:  list[int]      = field(default_factory=lambda: [32, 32, 32, 32])
    conv_k:   list[int]      = field(default_factory=lambda: [5, 3, 3, 3])
    conv_str: list[tuple[int,int]] = field(
        default_factory=lambda: [(2,1), (2,1), (2,2), (2,2)]
    )

    # fixed token grid after adaptive pool
    pool_F: int = 8      # freq patches
    pool_T: int = 64     # time patches  → 8·64 = 512 tokens

    # transformer
    layers:  int = 16
    d_model: int = 192
    n_heads: int = 6
    ff_mult: int = 4     # MLP expansion

# ──────────────────────────────────────
#  stem: conv stack
# ──────────────────────────────────────
class Stem(nn.Sequential):
    def __init__(self, cfg: BJConfig):
        layers, c_in = [], 1
        for c_out, k, (s_f, s_t) in zip(cfg.conv_ch, cfg.conv_k, cfg.conv_str):
            layers += [nn.Conv2d(c_in, c_out, k, stride=(s_f, s_t),
                                 padding=k // 2), nn.GELU()]
            c_in = c_out
        super().__init__(*layers)

# ──────────────────────────────────────
#  transformer block (full attention)
# ──────────────────────────────────────
class SDPABlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int):
        super().__init__()
        self.nh   = n_heads
        self.dh   = d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out  = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x):                     # (B, N, d)
        B, N, D = x.shape
        h = self.nh
        x = self.norm(x)
        x = rope(x)                           # 1-D rotary
        qkv = self.qkv(x).view(B, N, h, 3 * self.dh).chunk(3, dim=-1)
        q, k, v = (t.permute(0, 2, 1, 3) for t in qkv)  # (B,h,N,d_h)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        a = a.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        x = x + self.out(a)
        return x + self.ff(x)

# ──────────────────────────────────────
#  BirdJEPA encoder
# ──────────────────────────────────────
class BirdJEPA(nn.Module):
    """
    (B, 1, 128, 1000) log-mel → 512 tokens × d_model.
    """
    def __init__(self, cfg: BJConfig):
        super().__init__()
        self.cfg  = cfg
        self.stem = Stem(cfg)

        # channel dim after stem (computed once)
        with torch.no_grad():
            C = self.stem(torch.zeros(1, 1, cfg.n_mels, 1000)).shape[1]

        self.proj = nn.Linear(C, cfg.d_model, bias=False)
        self.proj.pool_F = cfg.pool_F    # expose grid dims for mask logic
        self.proj.pool_T = cfg.pool_T

        # transformer encoder
        self.encoder = nn.Sequential(
            *[SDPABlock(cfg.d_model, cfg.n_heads, cfg.ff_mult)
              for _ in range(cfg.layers)]
        )

    # ----------------------------------------------------------
    def forward(self, spec):                       # spec (B,1,F,T)
        z = self.stem(spec)                        # (B,C,F',T')
        z = F.adaptive_avg_pool2d(
                z, (self.cfg.pool_F, self.cfg.pool_T))      # (B,C,8,64)
        B, C, Fg, Tg = z.shape
        z = z.permute(0, 2, 3, 1).reshape(B, Fg * Tg, C)    # (B,512,C)
        return self.encoder(self.proj(z))                   # (B,512,d)
