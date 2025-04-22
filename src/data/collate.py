import torch, random, numpy as np

# ──────────────────────────────────────────────────────────────
#  “random rectangles until target volume” batch‑mask generator
# ──────────────────────────────────────────────────────────────
def _make_rect_mask(F: int, T: int, mask_p: float = .40,
                    max_tries: int = 1000) -> torch.BoolTensor:
    """Returns a (F,T) Bool mask with ~mask_p * F * T pixels = True."""
    target = int(mask_p * F * T)
    m = torch.zeros(F, T, dtype=torch.bool)
    covered = 0
    tries = 0
    while covered < target and tries < max_tries:
        tries += 1
        h = random.randint(1, F)          # height in freq bins
        w = random.randint(1, T)          # width  in time bins
        f0 = random.randint(0, F - h)
        t0 = random.randint(0, T - w)
        # number of NEW pixels this rectangle would add
        new = (~m[f0:f0+h, t0:t0+w]).sum().item()
        if new == 0:
            continue
        m[f0:f0+h, t0:t0+w] = True
        covered += new
    return m


def masking_collate(batch,
                    mask_p: float = 0.40, **_ignored):
    """
    Returns  (all torch.float32)
      ctx   : (B , F , T)  input  –  masked positions set to **0**
      tgt   : (B , F , T)  target – *only* masked positions keep value,
                           everything else **0**
      mask  : (B , F , T) boolean mask
    """
    # make sure everything is a torch tensor on CPU
    specs = [torch.as_tensor(s, dtype=torch.float32)
             if not torch.is_tensor(s) else s
             for (s, *_ ) in batch]            # list of (F,T) tensors

    # labels are the 2nd element of each triple
    labs = [torch.as_tensor(l, dtype=torch.long)
            if not torch.is_tensor(l) else l
            for (_, l, _ ) in batch]          # list of (T,) tensors
    y = torch.stack(labs)          # (B,T)
    # dataset returns (spec , labels , fname); we don't need labels here
    _, _, fnames = zip(*batch)
    x = torch.stack(specs)           # (B,F,T)

    B, F, T = x.shape
    # single (F,T) mask, then broadcast to (B,F,T)
    base = _make_rect_mask(F, T, mask_p).unsqueeze(0)    # (1,F,T)
    mask = base.repeat(B, 1, 1).contiguous()             # (B,F,T)

    # dense target / context -------------------------------------------
    ctx = x.clone()                 # (B , F , T)
    tgt = ctx.clone()
    ctx[mask] = 0                   # zero masked in ctx
    tgt[~mask] = 0                  # zero un‑masked in tgt

    return ctx, tgt, ctx, mask, y, fnames

# ────────────────────────────────────────────────────────────────
#  NEW rectangle‑volume mask
# ────────────────────────────────────────────────────────────────
def rect_mask_collate(
        batch,
        mask_vol: float = .80,       # fraction of *volume* to hide
        min_t: int = 10, max_t: int = 250,
        min_f: int = 8,  max_f: int = 128, **_ignored):
    """
    *One* rectangular mask is sampled and applied to the whole batch.
    Stops when ≥ mask_vol of the F×T area is covered (over‑masking is OK).
    Returns the usual tuple:
        full, target, context, labels, mask3, filenames
    Shapes follow masking_collate: (B,F,T) except mask = (B,T)
    """
    # make sure everything is a torch tensor on CPU
    specs = [torch.as_tensor(s, dtype=torch.float32)
             if not torch.is_tensor(s) else s
             for (s, *_ ) in batch]

    # labels are the 2nd element of each triple
    labs = [torch.as_tensor(l, dtype=torch.long)
            if not torch.is_tensor(l) else l
            for (_, l, _ ) in batch]          # list of (T,) tensors
    y = torch.stack(labs)          # (B,T)
    # dataset returns (spec , labels , fname); we don't need labels here
    _, _, fnames = zip(*batch)     # skip labels
    x = torch.stack(specs)           # (B,F,T)
    B, F, T = x.shape

    goal = int(mask_vol * F * T)
    m2 = torch.zeros(F, T, dtype=torch.bool)        # 2‑D mask, shared
    covered = 0
    while covered < goal:
        h = random.randint(min_f, min(max_f, F))
        w = random.randint(min_t, min(max_t, T))
        f0 = random.randint(0, F - h)
        t0 = random.randint(0, T - w)
        new = (~m2[f0:f0+h, t0:t0+w]).sum().item()
        if new == 0:            # completely overlapping – resample
            continue
        m2[f0:f0+h, t0:t0+w] = True
        covered += new

    mask3 = m2.unsqueeze(0).repeat(B, 1, 1).clone()      # (B,F,T)
    ctx   = x.masked_fill(mask3, 0)
    tgt   = x.masked_fill(~mask3, 0)
    return x, tgt, ctx, y, mask3, fnames
