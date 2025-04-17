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
            mask[b, t0:t0+blk] = True
            remain -= blk

    # expand mask to (B,F,T) so broadcasting is explicit
    mask3 = mask.unsqueeze(1).expand(-1, x.size(1), -1)   # (B,F,T)

    ctx = x.masked_fill(mask3,     0)   # zero OUT the masked bins
    tgt = x.masked_fill(~mask3,    0)   # zero OUT the context bins

    return x, tgt, ctx, y, mask, fnames
