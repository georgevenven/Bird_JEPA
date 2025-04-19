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