import torch, math, torch.nn as nn

# ------------------------------------------------------------------
class _SA(nn.Module):
    """self‑attention + FFN block (pre‑norm)"""
    def __init__(self, d_model, n_heads, ff_mult):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model))
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        h, _ = self.attn(self.n1(x), self.n1(x), self.n1(x), attn_mask=mask)
        x = x + h
        x = x + self.ff(self.n2(x))
        return x

# ------------------------------------------------------------------
def local_mask(T: int, radius: int, device):
    m = torch.ones(T, T, dtype=torch.bool, device=device)
    i = torch.arange(T, device=device)
    for d in range(-radius, radius + 1):
        j = (i + d).clamp_(0, T - 1)
        m[i, j] = False
    m = m & (~torch.eye(T, dtype=torch.bool, device=device))
    return m

def global_mask(T: int, stride: int, device):
    g = torch.arange(0, T, stride, device=device)
    m = torch.ones(T, T, dtype=torch.bool, device=device)
    m[g, :] = False
    m[:, g] = False
    m = m & (~torch.eye(T, dtype=torch.bool, device=device))
    return m

# ------------------------------------------------------------------
class GLBlock(nn.Module):
    """
    local or global attention block depending on `mode`
    mode ∊ {'local','global','full'}
    """
    def __init__(self, d_model, n_heads, ff_mult,
                 mode: str = "local",
                 radius: int | None = None,
                 stride: int | None = None):
        super().__init__()
        self.mode, self.radius, self.stride = mode, radius, stride
        self.core = _SA(d_model, n_heads, ff_mult)

    def _mask(self, T: int, device):
        # during torch.onnx.export we're inside a tracing context; bail to full attention
        if torch.jit.is_tracing():
            return None
        if self.mode == "local":
            return local_mask(T, self.radius or 4, device)
        if self.mode == "global":
            return global_mask(T, self.stride or 16, device)
        return None                        # full attention

    def forward(self, x):
        return self.core(x, self._mask(x.size(1), x.device))

# ──────────────────────────────────────────────
# sparse‑friendly blocks (no eye(), no fill_diagonal_)
# ──────────────────────────────────────────────
class _MHA(nn.Module):
    """thin wrapper so we don't repeat norm→attn→ffn boilerplate"""
    def __init__(self, d, h, ff_mult):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, h, batch_first=True)
        self.ff    = nn.Sequential(nn.LayerNorm(d),
                                   nn.Linear(d, d*ff_mult),
                                   nn.GELU(),
                                   nn.Linear(d*ff_mult, d))

    def forward(self, x, mask=None):
        h,_ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                        attn_mask=mask)
        return x + self.ff(x + h)

# ------- helpers -------------------------------------------------
def _sliding_mask(T, w, device):
    i = torch.arange(T, device=device)[:,None]
    j = torch.arange(T, device=device)[None,:]
    return ((j - i).abs() > w)          # False inside window

def _stripe_mask(T, stride, device):
    g = torch.arange(0, T, stride, device=device)
    mask = torch.ones(T, T, dtype=torch.bool, device=device)
    mask[g,:] = mask[:,g] = False
    return mask

# ------- blocks --------------------------------------------------
class LocalBlock(nn.Module):
    def __init__(self, d, h, ff_mult=4, window=64):
        super().__init__()
        self.core = _MHA(d, h, ff_mult)
        self.window = window
        self.register_buffer("mask", torch.empty(0), persistent=False)

    def forward(self, x):
        T = x.size(1)
        if self.mask.size(0) < T:                 # lazily grow mask
            self.mask = _sliding_mask(T, self.window, x.device)
        return self.core(x, self.mask[:T,:T])

class GlobalBlock(nn.Module):
    def __init__(self, d, h, ff_mult=4, stride=128):
        super().__init__()
        self.core = _MHA(d, h, ff_mult)
        self.stride = stride
        self.register_buffer("mask", torch.empty(0), persistent=False)

    def forward(self, x):
        T = x.size(1)
        if self.mask.size(0) < T:
            self.mask = _stripe_mask(T, self.stride, x.device)
        return self.core(x, self.mask[:T,:T])