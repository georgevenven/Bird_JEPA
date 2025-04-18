import torch, math, torch.nn as nn
import torch.nn.functional as F
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
from .rope import rope

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
class LGBlock(nn.Module):
    """
    windowed (local) attention + optional global tokens every `g`.
    HuggingFace LongformerSelfAttention exports to ONNX as
    com.microsoft.EfficientAttention (= sparse CUDA kernel in ORT).
    """
    def __init__(self, d_model=96, n_heads=4, window=50, g_stride=100, ff_mult=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.window = window
        self.g_stride = g_stride
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model*ff_mult), nn.GELU(),
            nn.Linear(d_model*ff_mult, d_model)
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.size()
        qkv = self.ln(x)
        qkv = rope(qkv)
        q = self.q_proj(qkv).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(qkv).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(qkv).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # Build band mask inline
        mask = local_mask(T, self.window, x.device)  # (T, T), bool
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        h = h.transpose(1, 2).contiguous().view(B, T, D)
        h = self.out_proj(h)
        x = x + h
        x = x + self.ff(self.ln(x))
        return x

# ──────────────────────────────────────────────
# sparse‑friendly blocks (no eye(), no fill_diagonal_)
# ──────────────────────────────────────────────
class _MHA(nn.Module):
    """thin wrapper so we don't repeat norm→attn→ffn boilerplate"""
    def __init__(self, d, h, ff_mult):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.n_heads = h
        self.d_head = d // h
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.ff    = nn.Sequential(nn.LayerNorm(d),
                                   nn.Linear(d, d*ff_mult),
                                   nn.GELU(),
                                   nn.Linear(d*ff_mult, d))

    def forward(self, x, mask=None):
        B, T, D = x.shape
        x_norm = self.norm1(x)
        x_norm = rope(x_norm)
        q = self.q_proj(x_norm).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        if mask is not None:
            # mask: (T, T) or (B, T, T) → (B, 1, T, T)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        h = h.transpose(1, 2).contiguous().view(B, T, D)
        h = self.out_proj(h)
        x = x + h
        x = x + self.ff(x + h)
        return x

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