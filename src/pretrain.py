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
from data.collate import masking_collate, rect_mask_collate
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import dataclasses
import torch.profiler
from torch.profiler import ProfilerAction

def make_collate(cfg):
    if cfg.rect_mask:
        return partial(rect_mask_collate,
                       mask_vol = 1 - cfg.keep_p,
                       min_t    = cfg.rect_min_t,
                       max_t    = cfg.rect_max_t,
                       min_f    = cfg.rect_min_f,
                       max_f    = cfg.rect_max_f)
    else:                                  # legacy time‑stripe mask
        return partial(masking_collate,
                       mask_p   = 1 - cfg.keep_p,
                       max_block= 50)

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
    cfg = BJConfig(pattern=args.attn_pattern)

    # legacy field needed by the data loader ─ use the context length
    cfg.mask_t = getattr(args, "context_length", 1000)   # default 1\u00a0000 frames
    cfg.keep_p    = args.keep_p
    cfg.rect_mask = args.rect_mask

    model = BirdJEPA(cfg).to(dev)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler= GradScaler()

    ds   = BirdSpectrogramDataset(args.train_dir, segment_len=cfg.mask_t, infinite=True)
    collate = make_collate(cfg)
    dl   = DataLoader(ds, batch_size=args.bs, shuffle=True,
                      collate_fn=collate,
                      num_workers=args.nw, pin_memory=True, drop_last=True)

    run = Path(args.run_dir)
    weights_dir = run / "weights"
    imgs_dir    = run / "imgs"
    cfg_path    = run / "config.json"
    for d in (run, weights_dir, imgs_dir):
        d.mkdir(parents=True, exist_ok=True)

    log = Tee(run / "train_log.txt")

    prof_steps = (1 + 1 + 3) * 2  # schedule length = 10
    step = 0; best = 9e9
    losses, ema_losses = [], []
    alpha = 0.1              # smoothing factor (0.1 ⇒ ~10‑step window)
    t0 = time.time()

    # --- Profiled steps ---
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(run / "tb_prof"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for batch in dl:
            full, tgt, ctx, _, mask, _ = batch     # ignore the dummy "labels" tensor
            full, tgt, ctx, mask = [x.to(dev) for x in (full, tgt, ctx, mask)]
            # ------------------------------------------------------------
            # ctx , tgt : (B ,F ,T)  from masking_collate
            ctx_spec = ctx.unsqueeze(1)             # (B ,1 ,F ,T)
            tgt_spec = tgt.unsqueeze(1)

            if hasattr(model.encoder, "inference_forward"):          # BirdJEPA
                ctx_repr, _ = model.encoder.inference_forward(ctx_spec)  # (B ,Ttok ,192)
                tgt_repr, _ = model.encoder.inference_forward(tgt_spec)
            else:                                                    # raw Sequential ckpt
                ctx_repr = model.encoder(ctx_spec)
                tgt_repr = model.encoder(tgt_spec)

            pred = model.predictor(ctx_repr)               # (B,T,H)

            loss = jepa_loss(ctx_repr, pred, tgt_repr, mask)
            if step % 200 == 0:
                if prof.current_action == ProfilerAction.NONE:
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
                ema_prev = ema_losses[-1] if ema_losses else loss.item()
                ema_losses.append(alpha * loss.item() + (1 - alpha) * ema_prev)

                if step % (args.log_every*10) == 0:
                    if prof.current_action == ProfilerAction.NONE:
                        plt.figure(figsize=(4,3))
                        plt.plot(losses, lw=.7)
                        plt.title('train loss')
                        plt.tight_layout()
                        plt.savefig(imgs_dir / 'loss_final.png', dpi=150)
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
                if prof.current_action == ProfilerAction.NONE:
                    with torch.no_grad():
                        debug_plot(step,
                                   full.cpu(), ctx.cpu(), mask.cpu(),
                                   pred.detach().cpu(), tgt_repr.detach().cpu(),
                                   imgs_dir)

            if prof.current_action == ProfilerAction.RECORD:
                prof.step()
            step += 1
            if step >= prof_steps or step >= args.steps:
                break

    # --- Regular training (no profiler) ---
    for epoch in range(999999):
        for batch in dl:
            full, tgt, ctx, _, mask, _ = batch     # ignore the dummy "labels" tensor
            if step >= args.steps:
                break
            full, tgt, ctx, mask = [x.to(dev) for x in (full, tgt, ctx, mask)]
            # ------------------------------------------------------------
            # ctx , tgt : (B ,F ,T)
            ctx_spec = ctx.unsqueeze(1)             # (B ,1 ,F ,T)
            tgt_spec = tgt.unsqueeze(1)

            if hasattr(model.encoder, "inference_forward"):          # BirdJEPA
                ctx_repr, _ = model.encoder.inference_forward(ctx_spec)  # (B ,Ttok ,192)
                tgt_repr, _ = model.encoder.inference_forward(tgt_spec)
            else:                                                    # raw Sequential ckpt
                ctx_repr = model.encoder(ctx_spec)
                tgt_repr = model.encoder(tgt_spec)

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
                ema_prev = ema_losses[-1] if ema_losses else loss.item()
                ema_losses.append(alpha * loss.item() + (1 - alpha) * ema_prev)

                if step % (args.log_every*10) == 0:     # draw every ~10 logs
                    plt.figure(figsize=(4,3))
                    plt.plot(losses, lw=.7)
                    plt.title('train loss')
                    plt.tight_layout()
                    plt.savefig(imgs_dir / 'loss_final.png', dpi=150)
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
            step += 1
        if step >= args.steps:
            break

    # ---------- final loss curve ----------
    plt.figure(figsize=(5,4))
    plt.plot(losses,      lw=.7, label='raw')
    plt.plot(ema_losses,  lw=1.2, ls='--', label=f'ema α={alpha}')
    plt.xlabel('step'); plt.ylabel('loss'); plt.legend()
    plt.tight_layout()
    plt.savefig(run / 'loss_curve.png', dpi=150)
    plt.close()

    # ---------------- dump config --------------------------
    cfg_dict = dataclasses.asdict(cfg)           # BJConfig → dict
    meta = {
        "best_ckpt": str(weights_dir / "best.pt"),
        "bjconfig":  cfg_dict,
        "train_args": vars(args)                  # cli flags
    }
    with open(cfg_path, "w") as f:
        json.dump(meta, f, indent=2)

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
    p.add_argument("--attn_pattern", default="local64,global128,local64,global128")
    p.add_argument("--keep_p",      type=float, default=0.20,
                  help="visible volume fraction (1‑keep_p is masked)")
    p.add_argument("--rect_mask",   action="store_true",
                  help="use rectangle masking instead of time stripes")
    args = p.parse_args()

    # Ensure train_dir and run_dir exist
    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.run_dir, exist_ok=True)

    pretrain(args)