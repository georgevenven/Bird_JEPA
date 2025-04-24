import torch, math

class Rope(torch.nn.Module):
    def __init__(self, max_t=1024, d_half=48, device='cuda'):
        super().__init__()
        theta = 10000 ** (-torch.arange(0, d_half, dtype=torch.float32) / d_half)
        pos = torch.arange(max_t, dtype=torch.float32)[:, None]
        ang = pos * theta                      # (T, d/2)
        self.register_buffer("sin", ang.sin())  # (T, d/2)
        self.register_buffer("cos", ang.cos())  # (T, d/2)

    def forward(self, x):                       # x (B,T,D)
        B, T, D = x.shape
        d2 = D // 2
        sin, cos = self.sin[:T], self.cos[:T]   # slice
        x1, x2 = x[..., :d2], x[..., d2:]
        return torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

def rope(x):
    """
    x : (B, T, d)   with d even
    returns same‑shape tensor with rotary pos enc applied to last dim
    """
    B,T,D = x.shape
    assert D % 2 == 0
    d2 = D//2
    theta = 1.0 / (10000 ** (torch.arange(0,d2//2,device=x.device)/ (d2//2)))
    pos   = torch.arange(T, device=x.device).unsqueeze(1)        # (T,1)
    angles= pos * theta                                          # (T,d/4)
    sin = angles.sin().repeat_interleave(2, dim=1)               # (T,d/2)
    cos = angles.cos().repeat_interleave(2, dim=1)               # (T,d/2)

    x1, x2 = x[..., :d2], x[..., d2:]
    out = torch.cat([x1*cos - x2*sin,
                     x1*sin + x2*cos], dim=-1)
    return out 

def rope_2d(x: torch.Tensor, shape: tuple[int, int], base: float = 10000.0):
    """
    factorised 2-D rotary PE (rows ⇢ first half dims, cols ⇢ second half).

    x      : (B, F*T, D)   with D % 4 == 0
    shape  : (F, T)  grid dims so  F*T == seq_len
    returns: (B, F*T, D)
    """
    B, N, D = x.shape
    F, T = shape
    assert F * T == N,  "shape mismatch"
    assert D % 4 == 0,  "need D divisible by 4"

    d_half    = D // 2
    d_quarter = d_half // 2
    device, dt = x.device, x.dtype

    idx  = torch.arange(N, device=device)
    row  = idx // T
    col  = idx %  T

    inv = (base ** (-torch.arange(0, d_quarter, device=device, dtype=dt) / d_quarter))
    ang_row = row[:, None] * inv
    ang_col = col[:, None] * inv

    sin_r = ang_row.sin()[None]   # (1,N,d/4)
    cos_r = ang_row.cos()[None]
    sin_c = ang_col.sin()[None]
    cos_c = ang_col.cos()[None]

    x_row, x_col = x[..., :d_half], x[..., d_half:]
    xr1, xr2 = x_row[..., :d_quarter], x_row[..., d_quarter:]
    xc1, xc2 = x_col[..., :d_quarter], x_col[..., d_quarter:]

    row_rot = torch.cat([xr1 * cos_r - xr2 * sin_r,
                         xr1 * sin_r + xr2 * cos_r], dim=-1)
    col_rot = torch.cat([xc1 * cos_c - xc2 * sin_c,
                         xc1 * sin_c + xc2 * cos_c], dim=-1)
    return torch.cat([row_rot, col_rot], dim=-1) 