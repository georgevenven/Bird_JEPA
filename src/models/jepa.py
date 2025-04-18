import torch.nn as nn, torch
from .bj_config import BJConfig
from .layers import LGBlock

# ------------------------------------------------------------------
class ConvStem(nn.Module):
    """simple 2‑D conv tower that keeps time resolution intact."""
    def __init__(self, cfg: BJConfig):
        super().__init__()
        layers, in_ch = [], 1
        for ch, k, s in zip(cfg.conv_ch, cfg.conv_k, cfg.conv_str):
            layers += [nn.Conv2d(in_ch, ch, k, stride=s, padding=k // 2),
                       nn.GELU()]
            in_ch = ch
        self.stem = nn.Sequential(*layers)
        self.proj = nn.Conv2d(in_ch, cfg.d_model, 1)

    def forward(self, x):                  # B,1,F,T
        z = self.stem(x)                   # B,C,f',T'
        return self.proj(z).flatten(2).transpose(1, 2)   # B,seq,d

# ------------------------------------------------------------------
class BirdJEPA(nn.Module):
    """
    • Conv stem → flatten freq → encoder (local/global)  
    • predictor = shallow transformer (default 1 block)  
    Forward returns (predicted, context) pair.
    """
    def __init__(self, cfg: BJConfig | None = None):
        super().__init__()
        cfg = cfg or BJConfig()
        self.cfg = cfg
        self.stem = ConvStem(cfg)

        enc = []
        for i in range(cfg.layers):
            global_flag = (i % cfg.global_every) == cfg.global_every - 1
            if global_flag:
                enc.append(LGBlock(cfg.d_model, cfg.n_heads, window=4, g_stride=max(1, cfg.n_global), ff_mult=cfg.ff_mult))
            else:
                enc.append(LGBlock(cfg.d_model, cfg.n_heads, window=4, g_stride=cfg.layers*10000, ff_mult=cfg.ff_mult))  # disables global tokens
        self.encoder = nn.Sequential(*enc)

        self.predictor = nn.Sequential(
            *[LGBlock(cfg.pred_dim, cfg.n_heads, window=4, g_stride=cfg.layers*10000, ff_mult=cfg.ff_mult) for _ in range(cfg.pred_layers)],
            nn.LayerNorm(cfg.pred_dim),
            nn.Linear(cfg.pred_dim, cfg.pred_dim))

    # -------------------------------------------------------------
    def forward(self, spec: torch.Tensor):
        """
        spec: (B,1,F,T)  – mel‑spectrogram with freq first  
        returns: pred, ctx  (both (B,seq,d))
        """
        z   = self.stem(spec)
        ctx = self.encoder(z)
        pred = self.predictor(ctx)
        return pred, ctx