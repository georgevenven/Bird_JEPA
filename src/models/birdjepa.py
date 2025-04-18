# src/models/birdjepa.py
# full rewrite – stem collapses freq, encoder pattern string builds
# local / global sparse‑aware blocks.  < 300 LoC

import math, torch, torch.nn as nn
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
    pattern: str = "local64,global128,local64,global128"
    d_model: int = 96
    n_heads: int = 4
    ff_mult: int = 4

    # predictor (unused here)
    pred_dim: int = 96
    pred_layers:int=1

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

    def forward(self, x):                       # B 1 F T
        z = self.stem(x)                        # B C F' T'
        z = z.mean(2).transpose(1,2)            # B T' (C)
        return self.proj(z)                     # B T' d

# ──────────────────────────────────────
#  sparse‑friendly blocks
# ──────────────────────────────────────
class _SA(nn.Module):
    def __init__(self, d, h, ff_mult):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, h, batch_first=True)
        self.ff  = nn.Sequential(nn.LayerNorm(d),
                                 nn.Linear(d, d*ff_mult),
                                 nn.GELU(),
                                 nn.Linear(d*ff_mult, d))
    def forward(self,x,mask=None):
        h,_ = self.att(self.n1(x),self.n1(x),self.n1(x),attn_mask=mask)
        return x + self.ff(x + h)

def _band_mask(T,w,device):
    i = torch.arange(T,device=device)[:,None]
    j = torch.arange(T,device=device)[None,:]
    return (j-i).abs() > w                      # bool, no eye()

def _stripe_mask(T,s,device):
    g = torch.arange(0,T,s,device=device)
    m = torch.ones(T,T,dtype=torch.bool,device=device)
    m[g,:] = m[:,g] = False
    return m

class LocalBlock(nn.Module):
    def __init__(self,d,h,ff,w): super().__init__(); self.w=w; self.core=_SA(d,h,ff); self.register_buffer('m',torch.empty(0),False)
    def forward(self,x):
        T=x.size(1)
        if self.m.size(0)<T: self.m=_band_mask(T,self.w,x.device)
        return self.core(x,self.m[:T,:T])

class GlobalBlock(nn.Module):
    def __init__(self,d,h,ff,s): super().__init__(); self.s=s; self.core=_SA(d,h,ff); self.register_buffer('m',torch.empty(0),False)
    def forward(self,x):
        T=x.size(1)
        if self.m.size(0)<T: self.m=_stripe_mask(T,self.s,x.device)
        return self.core(x,self.m[:T,:T])

# ──────────────────────────────────────
#  encoder builder from pattern string
# ──────────────────────────────────────
def build_encoder(cfg: BJConfig):
    blocks=[]
    for tok in cfg.pattern.split(','):
        if tok.startswith('local'):
            blocks.append(LocalBlock(cfg.d_model,cfg.n_heads,cfg.ff_mult,
                                     int(tok[5:])))
        elif tok.startswith('global'):
            blocks.append(GlobalBlock(cfg.d_model,cfg.n_heads,cfg.ff_mult,
                                      int(tok[6:])))
        else:
            blocks.append(_SA(cfg.d_model,cfg.n_heads,cfg.ff_mult))
    return nn.Sequential(*blocks)

# ──────────────────────────────────────
#  BirdJEPA (encoder‑only for finetune)
# ──────────────────────────────────────
class BirdJEPA(nn.Module):
    def __init__(self,cfg: BJConfig):
        super().__init__()
        self.stem = ConvStem(cfg)
        self.encoder = build_encoder(cfg)

    def forward(self,x):             # B 1 F T
        tok = self.stem(x)           # B T' d
        ctx = self.encoder(tok)      # B T' d
        return ctx, None             # keep signature
