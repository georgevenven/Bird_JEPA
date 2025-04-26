# src/models/birdjepa.py
# full rewrite – stem collapses freq, encoder pattern string builds
# local / global sparse‑aware blocks.  < 300 LoC

import math, torch, torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from .rope import rope, rope_2d
from .utils_misc import count_params

print(">> birdjepa loaded from", __file__)

# ──────────────────────────────────────
#  config
# ──────────────────────────────────────
@dataclass
class BJConfig:
    # stem
    n_mels:  int       = 128
    # four convs:  F halved on first two,  T halved on *all* four
    conv_ch:  list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    conv_k:   list[int] = field(default_factory=lambda: [5, 3, 3, 3])
    conv_str: list[tuple]=field(default_factory=lambda: [(2,1), (2,1), (2,1), (2,1)])

    # encoder
    layers: int = 8
    d_model: int = 192
    n_heads: int = 6
    ff_mult: int = 4
    pattern: str = None

    # predictor (unused here)
    pred_dim: int = 96
    pred_layers:int=1

    # ---------- self‑sup. masking -------------------------------------
    mask_t:      int   = 1000   # training segment length (time frames)
    keep_p:      float = 0.20   # *visible* volume fraction  (1‑keep_p is masked)
    rect_mask:   bool  = True   # use rectangle masking (else old time‑strip)
    rect_min_t:  int   = 10
    rect_max_t:  int   = 250
    rect_min_f:  int   = 8
    rect_max_f:  int   = 128

    def __post_init__(self):
        if self.pattern is None:
            self.pattern = ','.join(['local50,global100'] * (self.layers // 2))

# ──────────────────────────────────────
#  stem: conv stack, tracks Fp
# ──────────────────────────────────────
class Stem(nn.Sequential):
    def __init__(self, cfg: BJConfig):
        layers = []
        C_in = 1
        Fp = cfg.n_mels
        print(f"Initial input shape: (B, {C_in}, {Fp}, T)")
        for ch, k, (s_f, s_t) in zip(cfg.conv_ch, cfg.conv_k, cfg.conv_str):
            layers += [nn.Conv2d(C_in, ch, k, stride=(s_f, s_t), padding=k//2), nn.GELU()]
            print(f"After conv {len(layers)//2}: (B, {ch}, {Fp//s_f}, T//{s_t})")
            C_in = ch
            Fp //= s_f
        super().__init__(*layers)
        cfg.Fp = Fp  # remember final freq bins
        print(f"Final output shape: (B, {C_in}, {Fp}, T//{2**len(cfg.conv_str)})")

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
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, grid_shape):
        B, T, D = x.shape
        x_norm = self.norm1(x)
        x_norm = rope_2d(x_norm, grid_shape)
        q = self.q_proj(x_norm).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        h = h.transpose(1, 2).contiguous().view(B, T, D)
        h = self.out_proj(h)
        x = x + h
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

# ──────────────────────────────────────
#  encoder builder from pattern string
# ──────────────────────────────────────
def build_encoder(cfg: BJConfig):
    return nn.ModuleList([SDPABlock(cfg.d_model, cfg.n_heads, cfg.ff_mult)
                         for _ in range(cfg.layers)])

# ──────────────────────────────────────
#  BirdJEPA (encoder‑only for finetune)
# ──────────────────────────────────────
class BirdJEPA(nn.Module):
    def __init__(self, cfg: BJConfig):
        super().__init__()
        self.cfg = cfg

        # ----- conv stem: (B,1,F,T) -> (B,C,F',T') ------------------------
        self.stem = nn.Sequential(
            *[nn.Conv2d(in_ch, out_ch, k, stride=s, padding=k//2)
              for in_ch, out_ch, k, s in zip(
                    [1]+cfg.conv_ch[:-1], cfg.conv_ch,
                    cfg.conv_k,           cfg.conv_str)]
        )

        # ----- figure out flattened dim dynamically -----------------------
        with torch.no_grad():
            dummy = torch.zeros(1, 1, cfg.n_mels, 256)     # 5 s clip → 256 frames
            C, Fp, Tp = self.stem(dummy).shape[1:]
        self.proj = nn.Linear(C, cfg.d_model, bias=False)  # per-patch linear

        # ----- transformer encoder (unchanged) ----------------------------
        self.core = build_encoder(cfg)         # ModuleList
        self.encoder = lambda spec: self(spec) # plain python func, not a sub-module
        self._print_par_count()
        # ------------------------------------------------------------------

    def forward(self, spec):                 # spec (B,1,F,T)
        z = self.stem(spec)
        B, C, Fp, Tp = z.shape                 # (32,16)
        z = z.permute(0, 2, 3, 1).reshape(B, Fp * Tp, C)
        x = self.proj(z)

        for blk in self.core:                  # ← iterate manually
            x = blk(x, (Fp, Tp))               # pass (F′,T′) for 2-D RoPE
        return x
    
    def _print_par_count(self):
        m = count_params(self)
        print(f"[birdjepa] {m:.2f}M parameters")

# ──────────────────────────────────────
#  Predictor
# ──────────────────────────────────────
class Predictor(nn.Module):
    """
    shallow MLP g(·) with rotary PE baked in.
    optionally consumes the boolean time‑mask so you can
    condition on "which tokens were hidden" later.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model)
        )

    def forward(self, x, grid_shape, mask: torch.Tensor | None = None):
        # x : (B, T, d)
        x = rope_2d(self.norm(x), grid_shape)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return self.mlp(x)