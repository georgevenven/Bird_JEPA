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
from models.birdjepa import BirdJEPA, BJConfig
from data.bird_datasets import BirdSpectrogramDataset
from data.collate import masking_collate
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

def make_collate(mask_p, max_block=50):
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

    run = Path(args.run_dir)
    weights_dir = run / "weights"
    imgs_dir    = run / "imgs"
    for d in (run, weights_dir, imgs_dir):
        d.mkdir(parents=True, exist_ok=True)

    log = Tee(run / "train_log.txt")

    step = 0; best = 9e9
    losses, ema = deque(maxlen=200), None        # rolling reservoir for  plotting
    t0 = time.time()
    for epoch in range(999999):
        for full,tgt,ctx,_,mask,_ in dl:
            full,tgt,ctx,mask = [x.to(dev) for x in (full,tgt,ctx,mask)]
            # ------------------------------------------------------------
            # encode context and target
            ctx_spec = ctx.unsqueeze(1).transpose(2, 3)  # (B,1,F,T)
            tgt_spec = tgt.unsqueeze(1).transpose(2, 3)  # (B,1,F,T)
            ctx_seq = model.stem(ctx_spec)               # (B,seq,192)
            tgt_seq = model.stem(tgt_spec)               # (B,seq,192)
            ctx_repr = model.encoder(ctx_seq)
            with torch.no_grad():
                tgt_repr = model.ema_encoder(tgt_seq)

            pred = model.predictor(ctx_repr)               # (B,T,H)

            loss = jepa_loss(ctx_repr, pred, tgt_repr, mask)
            if step % 200 == 0:
                debug_plot(step, full.cpu(), ctx.cpu(), mask.cpu(),
                           pred.detach().cpu(), tgt_repr.detach().cpu(),
                           imgs_dir)
            # ------------------------------------------------------------

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            _update_ema(model)

            if step % args.log_every == 0:
                fps = args.log_every*args.bs / (time.time()-t0); t0=time.time()
                log(f"step {step:06d}  loss {loss.item():.4f}  {fps:.1f} samp/s")

                # ---- loss tracking ----
                losses.append(loss.item())
                ema = loss.item() if ema is None else 0.99*ema + 0.01*loss.item()

                if step % (args.log_every*10) == 0:     # draw every ~10 logs
                    plt.figure(figsize=(4,3))
                    plt.plot(list(losses), lw=.7, label='raw')
                    plt.axhline(ema, color='k', ls='--', lw=1, label='ema')
                    plt.legend(); plt.tight_layout()
                    plt.savefig(imgs_dir / f"loss_{step:06d}.png")
                    plt.close()

                # ------------------------------------------------------
                # 1) save *every* checkpoint
                # ------------------------------------------------------
                ckpt_path = weights_dir / f"step_{step:06d}.pt"
                torch.save(model.state_dict(), ckpt_path)

                # keep track of the best loss for convenience
                if loss.item() < best:
                    best = loss.item()
                    torch.save(model.state_dict(), weights_dir / "best.pt")

                # ------------------------------------------------------
                # 2) visual diag → <run_dir>/imgs/
                # ------------------------------------------------------
                with torch.no_grad():
                    debug_plot(step,
                               full.cpu(), ctx.cpu(), mask.cpu(),
                               pred.detach().cpu(), tgt_repr.detach().cpu(),
                               imgs_dir)

            if step >= args.steps: break
            step += 1
        if step >= args.steps: break

    log("done. best={:.4f}".format(best)); log.close()

def _update_ema(model):
    with torch.no_grad():
        for p_online, p_ema in zip(model.encoder.parameters(),
                                   model.ema_encoder.parameters()):
            p_ema.data.lerp_(p_online.data, 1 - model._ema_tau)

def unwrap_seq(h, F_orig):
    """
    (seq,d)  ->  (F',T') by:
      1) pick a divisor of seq that's closest to F_orig/8
      2) reshape using that divisor
    """
    S = h.size(0)
    target = max(1, F_orig // 8)          # ≈ what we expect
    # scan outward until we find a divisor
    for off in range(0, target):          # enough wiggle room
        for cand in (target - off, target + off):
            if cand > 0 and S % cand == 0:
                Fp = cand
                Tp = S // cand
                return h.mean(-1)[:S].reshape(Fp, Tp)
    raise ValueError(f"can't factor seq={S}")

@torch.no_grad()
def debug_plot(step, full, ctx, mask, pred, tgt, run_dir):
    """run_dir is now the imgs folder we pass in"""
    raw   = full[0].cpu().numpy()        # (F,T)
    mctx  = ctx[0].cpu().numpy()         # (F,T)
    lossm = mask[0].any(0).cpu().numpy() # (T,) → for overlay dots
    F = raw.shape[0]
    p_img = unwrap_seq(pred[0], F)      # just the first sample
    t_img = unwrap_seq(tgt[0],  F)
    diff  = (p_img - t_img) ** 2

    fig, ax = plt.subplots(2,2, figsize=(10,8))
    ax = ax.ravel()
    ax[0].imshow(raw,  aspect='auto', origin='lower'); ax[0].set_title('orig')
    # --- overlay mask as alpha-blended red on masked ctx ---
    ax[1].imshow(mctx, aspect='auto', origin='lower'); ax[1].set_title('masked ctx')
    mask_overlay = mask[0].cpu().numpy().astype(float)  # (F,T), 1 where masked
    red = np.zeros((*mask_overlay.shape, 4), dtype=float)
    red[..., 0] = 1.0  # Red channel
    red[..., 3] = 0.4 * mask_overlay  # Alpha channel, 0.4 where masked
    ax[1].imshow(red, aspect='auto', origin='lower')
    # ------------------------------------------------------
    ax[2].imshow(p_img,aspect='auto', origin='lower'); ax[2].set_title('pred')
    ax[3].imshow(diff, aspect='auto', origin='lower'); ax[3].set_title('sq-error')
    for a in ax: a.axis('off')
    plt.tight_layout()
    fig.savefig(run_dir / f"viz_{step:06d}.png")
    plt.close(fig)

# ------------- cli ----------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--run_dir",   default="runs/pretrain")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--bs",    type=int, default=64)
    p.add_argument("--lr",    type=float, default=1e-4)
    p.add_argument("--nw",    type=int, default=4)
    p.add_argument("--log_every", type=int, default=150)
    args = p.parse_args()
    pretrain(args)