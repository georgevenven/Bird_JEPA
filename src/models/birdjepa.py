# src/models/birdjepa.py
# full rewrite – stem collapses freq, encoder pattern string builds
# local / global sparse‑aware blocks.  < 300 LoC

import math, torch, torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

# ──────────────────────────────────────
#  config
# ──────────────────────────────────────
@dataclass
class BJConfig:
    # stem
    n_mels:  int       = 513
    conv_ch: list[int] = field(default_factory=lambda: [32, 32, 32])
    conv_k:  list[int] = field(default_factory=lambda: [5, 3, 3])
    conv_str:list[tuple]=field(default_factory=lambda: [(2,1),(2,2),(2,2)])

    # encoder
    layers: int = 16
    d_model: int = 192
    n_heads: int = 6
    ff_mult: int = 4
    pattern: str = None

    # predictor (unused here)
    pred_dim: int = 96
    pred_layers:int=1

    def __post_init__(self):
        if self.pattern is None:
            self.pattern = ','.join(['local50,global100'] * (self.layers // 2))

# ──────────────────────────────────────
#  stem: conv → (mean freq) → Linear
# ──────────────────────────────────────
class ConvStem(nn.Module):
    def __init__(self, cfg: BJConfig):
        super().__init__()
        layers, in_ch = [], 1
        for ch, k, s in zip(cfg.conv_ch, cfg.conv_k, cfg.conv_str):
            layers += [nn.Conv2d(in_ch, ch, k, stride=s, padding=k//2), nn.GELU()]
            in_ch = ch
        self.stem = nn.Sequential(*layers)
        self.proj = nn.LazyLinear(cfg.d_model)
        self.freq_collapse = nn.LazyLinear(cfg.d_model, bias=False)

    def forward(self, x):                       # B 1 F T
        z = self.stem(x)                        # B C F' T'
        z = self.proj(z)                        # B d F' T'
        f_prime = z.size(2)
        z = z.flatten(2, 3).transpose(1, 2)     # B T' (d·F')
        return self.freq_collapse(z)            # B T' d

# ──────────────────────────────────────
#  SDPA-based attention blocks
# ──────────────────────────────────────
class SDPABlock(nn.Module):
    def __init__(self, d, h, ff_mult):
        super().__init__()
        self.n_heads = h
        self.d_head = d // h
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d*ff_mult),
            nn.GELU(),
            nn.Linear(d*ff_mult, d)
        )

    def forward(self, x):
        B, T, D = x.shape
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        h = h.transpose(1, 2).contiguous().view(B, T, D)
        h = self.out_proj(h)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x

# ──────────────────────────────────────
#  encoder builder from pattern string
# ──────────────────────────────────────
def build_encoder(cfg: BJConfig):
    blocks=[]
    for _ in range(cfg.layers):
        blocks.append(SDPABlock(cfg.d_model, cfg.n_heads, cfg.ff_mult))
    return nn.Sequential(*blocks)

# ──────────────────────────────────────
#  BirdJEPA (encoder‑only for finetune)
# ──────────────────────────────────────
class BirdJEPA(nn.Module):
    def __init__(self, cfg: BJConfig):
        super().__init__()
        self.stem = ConvStem(cfg)
        self.proj = nn.Linear(
            cfg.conv_ch[-1] * (cfg.n_mels // 8),   # 32 * 64 = 2048
            cfg.d_model, bias=False
        )
        self.encoder = build_encoder(cfg)

    def forward(self, spec):          # spec (B, F, T)
        spec = spec.unsqueeze(1).transpose(2, 3)      # (B,1,T,F) → (B,1,F,T)
        z = self.stem(spec)                           # (B,C,F',T')
        print("STEM :", z.shape)
        B, C, Fp, Tp = z.shape
        z = z.permute(0, 3, 1, 2).reshape(B, Tp, C*Fp) # (B,T',C*F')
        z = self.proj(z)                              # (B,T',d_model)
        print("PROJ :", z.shape)
        z = self.encoder(z)
        return z
