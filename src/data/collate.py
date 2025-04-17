import torch, random

def masking_collate(batch,
                    mask_p: float = 0.75,
                    max_block: int = 10):
    """
    Returns:
        full, target, context, labels, mask, filenames
    Shapes:  (B,F,T) except mask = (B,T)
    """
    specs, labs, fnames = zip(*batch)
    x = torch.stack(specs)           # (B,F,T)
    y = torch.stack(labs)            # (B,T)

    B, F, T = x.shape
    mask = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        remain = int(mask_p * T)
        while remain:
            blk = min(random.randint(1, max_block), remain)
            t0  = random.randint(0, T - blk)
            mask[b, t0 : t0 + blk] = True
            remain -= blk

    # broadcast *with own storage*  (no view/expand aliasing)
    mask3 = mask.unsqueeze(1).repeat(1, F, 1).clone()             # (B,F,T)

    # randomly keep half of the freqs inside the time blocks
    freq_keep = torch.rand_like(mask3, dtype=torch.float) > 0.5
    mask3 &= freq_keep                                            # still bool

    ctx = x.masked_fill(mask3,     0)   # zero OUT the masked bins
    tgt = x.masked_fill(~mask3,    0)   # zero OUT the context bins

    return x, tgt, ctx, y, mask3, fnames
