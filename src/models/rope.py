import torch, math

def rope(x):
    """
    x : (B, T, d)   with d even
    returns same‑shape tensor with rotary pos enc applied to last dim
    """
    B,T,d = x.shape
    d2 = d // 2
    theta = 1.0 / (10000 ** (torch.arange(0,d2,2,device=x.device)/d2))
    pos   = torch.arange(T, device=x.device).unsqueeze(1)      # (T,1)
    angles= pos * theta                                        # (T,d/2)
    sin, cos = angles.sin(), angles.cos()                      # (T,d/2)

    # interleave even/odd dims
    x1, x2 = x[..., :d2], x[..., d2:]
    x_rot = torch.stack([-x2, x1], dim=-1).reshape(B,T,d2)     # 90° rotation
    out   = torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
    return out 