"""
self‑supervised pre‑training for BirdJEPA
 * JEPA loss = MSE on masked time bins
 * EMA target encoder
 * logs to terminal and <run_dir>/train_log.txt
"""

import argparse, json, time, math, os
from pathlib import Path
import torch, torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from models.jepa import BirdJEPA, BJConfig
from data.bird_datasets import BirdSpectrogramDataset
from data.collate import masking_collate
from functools import partial

def make_collate(mask_p, max_block=10):
    return partial(masking_collate, mask_p=mask_p, max_block=max_block)

# ------------- utils --------------------------------------------------------
class Tee:
    def __init__(self, fn):
        self.file = open(fn, "a", buffering=1)
    def __call__(self, *msg):
        txt = " ".join(str(m) for m in msg)
        print(txt); self.file.write(txt+"\n")
    def close(self): self.file.close()

# ------------- JEPA loss ----------------------------------------------------
def jepa_loss(ctx, pred, tgt, mask):
    # ctx not used but returned for monitoring
    # mask comes in as (B,F,T). After conv‑stem + flatten we lost F
    # → just squeeze it away:
    mask2d = mask.any(dim=1)                        # (B,T)
    # ---------- make sure all three share an identical T --------------
    T = min(pred.size(1), tgt.size(1), mask2d.size(1))
    pred    = pred[:, :T]           # (B,T,d)
    tgt     = tgt[:,  :T]           # (B,T,d)
    mask2d  = mask2d[:, :T]         # (B,T)

    diff = (pred - tgt) ** 2 * mask2d.unsqueeze(-1)   # (B,T,d)
    loss = diff.sum() / (mask2d.sum() * pred.size(-1) + 1e-8)
    return loss

# ------------- pre‑train ----------------------------------------------------
def pretrain(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = BJConfig()
    model = BirdJEPA(cfg).to(dev)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler= GradScaler()

    ds   = BirdSpectrogramDataset(args.train_dir, segment_len=cfg.mask_t, infinite=True)
    collate = make_collate(1-cfg.keep_p)
    dl   = DataLoader(ds, batch_size=args.bs, shuffle=True,
                      collate_fn=collate,
                      num_workers=args.nw, pin_memory=True, drop_last=True)

    run = Path(args.run_dir); run.mkdir(parents=True, exist_ok=True)
    log = Tee(run/"train_log.txt")

    step = 0; best = 9e9
    t0 = time.time()
    for epoch in range(999999):
        for full,tgt,ctx,_,mask,_ in dl:
            full,tgt,ctx,mask = [x.to(dev) for x in (full,tgt,ctx,mask)]
            pred,_ = model(ctx.unsqueeze(1))       # (B,T,d)
            # Project the target to match pred's embedding dimension
            stem_out = model.stem(full.unsqueeze(1).transpose(2, 3))[0]  # (B,T,flat)
            # project the stem features to the same hidden_dim as `pred`
            # this works for BOTH legacy and new BirdJEPA variants
            if   hasattr(model, "context_encoder"):          # new code‑base
                tgt = model.context_encoder.input_proj(stem_out)            # (B,T,H)
            elif hasattr(model, "encoder"):                  # legacy (this file)
                tgt = model.encoder.input_proj(stem_out)                     # (B,T,H)
            else:                                            # last‑resort: make one
                in_dim  = stem_out.size(-1)
                hid_dim = pred.size(-1)
                model._tmp_proj = nn.Linear(in_dim, hid_dim).to(stem_out.device)
                tgt = model._tmp_proj(stem_out)
            loss = jepa_loss(ctx, pred, tgt, mask)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()

            if step % args.log_every == 0:
                fps = args.log_every*args.bs / (time.time()-t0); t0=time.time()
                log(f"step {step:06d}  loss {loss.item():.4f}  {fps:.1f} samp/s")

                if loss.item() < best:
                    best = loss.item()
                    torch.save(model.state_dict(), run/"best.pt")

            if step >= args.steps: break
            step += 1
        if step >= args.steps: break

    log("done. best={:.4f}".format(best)); log.close()

# ------------- cli ----------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--run_dir",   default="runs/pretrain")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--bs",    type=int, default=64)
    p.add_argument("--lr",    type=float, default=1e-4)
    p.add_argument("--nw",    type=int, default=4)
    p.add_argument("--log_every", type=int, default=50)
    args = p.parse_args()
    pretrain(args)