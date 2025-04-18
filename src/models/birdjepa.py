import torch.nn as nn, torch
from .bj_config import BJConfig
from .layers import GLBlock, LocalBlock, GlobalBlock
import copy

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
def _make_encoder(cfg: BJConfig, pattern: str):
    """
    pattern example: "local64,global128,local64,global128"
    """
    enc = nn.ModuleList()
    for token in pattern.split(','):
        if token.startswith('local'):
            w = int(token[5:])
            enc.append(LocalBlock(cfg.d_model, cfg.n_heads,
                                  cfg.ff_mult, window=w))
        elif token.startswith('global'):
            s = int(token[6:])
            enc.append(GlobalBlock(cfg.d_model, cfg.n_heads,
                                   cfg.ff_mult, stride=s))
        else:                                   # fallback full
            enc.append(GLBlock(cfg.d_model, cfg.n_heads, cfg.ff_mult,
                               "full"))
    return nn.Sequential(*enc)

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

        self.encoder = _make_encoder(cfg, cfg.pattern)

        self.ema_encoder = copy.deepcopy(self.encoder)
        for p in self.ema_encoder.parameters():
            p.requires_grad_(False)
        self._ema_tau = 0.999   # decay; tweak as you like

        self.predictor = nn.Sequential(
            *[GLBlock(cfg.pred_dim, cfg.n_heads, cfg.ff_mult,
                      "local", radius=4) for _ in range(cfg.pred_layers)],
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

    def project_to_hidden(self, x):
        return self.encoder[0].core.n1(x) if hasattr(self.encoder[0].core, 'n1') else x  # fallback, can be adjusted 