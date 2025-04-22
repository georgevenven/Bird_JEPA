# ──────────────────────────────────────────────────────────────────────────────
# src/pretrain.py      • BirdJEPA encoder self‑supervised training (rect‑mask)
# ──────────────────────────────────────────────────────────────────────────────
import argparse, json, time, os, shutil, uuid, datetime as dt
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.multiprocessing
import torch.nn.functional as tF        # avoid name clash
import random

from models          import BJConfig, Predictor
from utils           import load_pretrained_encoder
from data.bird_datasets import TorchSpecDataset

# ---------------------------------------------------------------
AMP = torch.cuda.is_available()
if AMP:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True

# ╭───────────────────────────────────────────────────────────────╮
# │ Trainer                                                     │
# ╰───────────────────────────────────────────────────────────────╯
class Trainer:
    def __init__(self, args):
        self.args = args
        torch.multiprocessing.set_sharing_strategy("file_system")

        # ── data ────────────────────────────────────────────────
        self.ds  = TorchSpecDataset(args.train_dir, segment_len=args.context_length)
        self.val = TorchSpecDataset(args.val_dir  or args.train_dir,
                                    segment_len=args.context_length,
                                    infinite=False)
        self.train_dl = torch.utils.data.DataLoader(self.ds,  batch_size=args.bs,
                                                    shuffle=True,  num_workers=4,
                                                    pin_memory=True,  drop_last=True)
        self.val_dl   = torch.utils.data.DataLoader(self.val, batch_size=args.bs,
                                                    shuffle=False, num_workers=0)

        # ── model  (encoder + predictor) ────────────────────────
        cfg   = BJConfig(d_model=args.d_model, pattern=args.attn_pattern)
        self.enc = load_pretrained_encoder(cfg, None)          # start from scratch
        self.pred = Predictor(cfg.d_model)
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enc.to(self.dev); self.pred.to(self.dev)

        # belt-and-suspenders: always provide inference_forward
        if not hasattr(self.enc, "inference_forward"):
            self.enc.inference_forward = lambda x: (self.enc(x), None)

        # ── opt / sched ────────────────────────────────────────
        self.opt   = torch.optim.AdamW(list(self.enc.parameters())+
                                       list(self.pred.parameters()),
                                       lr=args.lr, weight_decay=1e-2)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.opt, T_max=args.steps, eta_min=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=AMP)
        self.crit   = nn.MSELoss()

        # ── run dir / logging ──────────────────────────────────
        self.run_dir = Path(args.run_dir); self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir/'weights').mkdir(exist_ok=True)
        (self.run_dir/'config.json').write_text(json.dumps(vars(args), indent=2))

        # track
        self.best_val = 9e9; self.bad_evals = 0
        self.hist = {'step':[], 'train':[], 'val':[]}

        self._eval_counter = 0

    # -----------------------------------------------------------------
    # ──────────────────────────────────────────────────────────────
    #  rectangle mask on raw (F,T) spectrogram → pooled to tokens
    # ──────────────────────────────────────────────────────────────
    def _rect_mask(self, F_bins:int, T_bins:int, mask_p:float=.40,
                   device='cpu'):
        """returns (F,T) bool mask"""
        target = int(mask_p * F_bins * T_bins)
        m = torch.zeros(F_bins, T_bins, dtype=torch.bool, device=device)
        covered = 0
        tries = 0
        while covered < target and tries < 1000:
            tries += 1
            h = random.randint(1, F_bins)
            w = random.randint(1, T_bins)
            f0 = random.randint(0, F_bins - h)
            t0 = random.randint(0, T_bins - w)
            new = (~m[f0:f0+h, t0:t0+w]).sum().item()
            if new == 0:
                continue
            m[f0:f0+h, t0:t0+w] = True
            covered += new
        return m

    # -----------------------------------------------------------------
    def _step(self, spec):
        spec = spec.to(self.dev)                  # (B,F,T)
        B,F_bins,T_bins = spec.shape

        # -------------- draw one batch mask -------------------
        base = self._rect_mask(F_bins, T_bins,
                               self.args.mask_ratio,
                               device=spec.device)        # (F,T)
        mask3 = base.unsqueeze(0).repeat(B,1,1)            # (B,F,T)

        # zero masked bins (stop‑grad) ------------------------
        spec_masked = spec.clone()
        spec_masked[mask3] = 0.0

        with torch.cuda.amp.autocast(enabled=AMP):
            spec_masked = spec_masked.unsqueeze(1)                  # (B,1,F,T)
            if hasattr(self.enc, "inference_forward"):
                z, _ = self.enc.inference_forward(spec_masked)
            else:
                z = self.enc(spec_masked)
            B, T_tok, D = z.shape        # T_tok = T_bins//4 (conv stride)

            # max‑pool mask3 over freq, then over 4‑frame time windows
            time_mask = mask3.any(1).float().unsqueeze(1)           # (B,1,T_bins)
            pooled = F.max_pool1d(time_mask, kernel_size=4, stride=4)
            mask_tok = pooled.squeeze(1).bool()                     # (B,T_tok)

            z_ctx = z.clone()
            z_ctx[mask_tok] = 0
            pred = self.pred(z_ctx)                                 # (B,T,d)

            loss = self.crit(pred[mask_tok], z[mask_tok])

        self.opt.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(list(self.enc.parameters())+
                                       list(self.pred.parameters()), 1.0)
        self.scaler.step(self.opt); self.scaler.update()
        # ── optional viz ───────────────────────────────────
        if self.args.viz_every and (self.args.viz_every > 0) and (hasattr(self, 'step') and (self.step % self.args.viz_every == 0)):
            with torch.no_grad():
                s0 = spec[0].cpu()              # (F,T)
                m0 = mask3[0].cpu()              # (F,T)
                p0 = pred[0].cpu()              # (T,D)
                z0 = z[0].cpu()                 # (T,D)
            _viz_triptych(self.run_dir, self.step, s0, m0, p0, z0)
        return float(loss.item())

    # -----------------------------------------------------------------
    @torch.no_grad()
    def _eval(self):
        self.enc.eval(); self.pred.eval()
        tot = 0.0; n = 0
        for spec, *_ in self.val_dl:
            spec = spec.to(self.dev)
            with torch.cuda.amp.autocast(enabled=AMP):
                s = spec.unsqueeze(1)                 # (B,1,F,T)
                if hasattr(self.enc, "inference_forward"):
                    z, _ = self.enc.inference_forward(s)
                else:                                 # vanilla Sequential
                    z = self.enc(s)
                B,T,D = z.shape
                mask = self._rect_mask(D, T, self.args.mask_ratio, device=z.device)
                z_ctx = z.clone(); z_ctx[mask] = 0
                pred = self.pred(z_ctx)
                tot += self.crit(pred[mask], z[mask]).item()*B
                n   += B
            if self.args.viz_every and (self._eval_counter % self.args.viz_every == 0):
                _viz_triptych(self.run_dir, self._eval_counter, spec[0,0].cpu(),
                              mask[0].cpu(), pred[0].cpu(), z[0].cpu())
            self._eval_counter += 1
        self.enc.train(); self.pred.train()
        return tot/n

    # -----------------------------------------------------------------
    def train(self):
        dl_iter = iter(self.train_dl); step = 0
        first_step = True
        while step < self.args.steps:
            try:       spec, *_ = next(dl_iter)
            except StopIteration:
                dl_iter = iter(self.train_dl); continue

            step += 1
            self.step = step
            tr_loss = self._step(spec)
            if first_step:
                first_step = False            # scheduler already advanced in _step()
            else:
                if step > 1:           # skip the misleading first tick
                    self.sched.step()

            if step % self.args.log_every == 0:
                val_loss = self._eval()
                print(f"step {step:07d}  train {tr_loss:.4f}  val {val_loss:.4f}")
                self.hist['step'].append(step)
                self.hist['train'].append(tr_loss)
                self.hist['val'].append(val_loss)

                if val_loss < self.best_val - 1e-4:
                    self.best_val = val_loss; self.bad_evals = 0
                    torch.save({'enc': self.enc.state_dict(),
                                'pred': self.pred.state_dict()},
                               self.run_dir/'weights/best.pt')
                else:
                    self.bad_evals += 1
                if self.bad_evals >= self.args.early_stop:
                    print("early stop")
                    break

        # plots ---------------------------------------------------
        df = pd.DataFrame(self.hist); df.to_csv(self.run_dir/'metrics.csv', index=False)
        plt.figure(); plt.plot(df.step, df.train, label='train'); plt.plot(df.step, df.val, label='val')
        plt.legend(); plt.xlabel('step'); plt.ylabel('loss')
        plt.tight_layout(); plt.savefig(self.run_dir/'loss_curve.png', dpi=150)

# ───────────────────────────── viz util ──────────────────────────────
def _viz_triptych(run_dir, step, spec, mask, pred, target):
    """
    spec  : (1,F,T)  float32 cpu
    mask  : (F,T)   bool
    pred  : (T,D)    torch
    target: (T,D)    torch
    """
    import matplotlib.pyplot as plt, numpy as np
    spec = spec.squeeze(0)                  # (F,T_full)
    F_bins, T_full = spec.shape
    # align spec's time axis to mask length  ---------------------------
    if T_full != mask.shape[1]:
        spec_ds = tF.interpolate(spec.unsqueeze(0).unsqueeze(0),
                                 size=mask.shape[1], mode="nearest"
                                 ).squeeze(0).squeeze(0)          # (F,T′)
    else:
        spec_ds = spec

    spec_masked = spec_ds.clone()
    spec_masked[mask] = 0.0          # zero wherever mask == True (2‑D)

    err = (pred - target).abs().mean(-1).cpu().numpy()   # (T,)
    err_img = np.tile(err, (F_bins, 1))                  # broadcast → F,T′

    fig,ax = plt.subplots(1,3,figsize=(9,3),dpi=120)
    kw = dict(aspect='auto',origin='lower',cmap='magma')
    ax[0].imshow(spec, **kw);          ax[0].set_title("input")
    ax[1].imshow(spec_masked, **kw);   ax[1].set_title("masked")
    ax[2].imshow(err_img, **kw);       ax[2].set_title("|pred‑ctx|")

    for a in ax: a.axis('off')
    out = run_dir / f"viz_{step:07d}.png"
    fig.tight_layout(); fig.savefig(out); plt.close(fig)

# ╭───────────────────────────────────────────────────────────────╮
# │ CLI                                                          │
# ╰───────────────────────────────────────────────────────────────╯
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', required=True)
    p.add_argument('--val_dir')
    p.add_argument('--run_dir',  default='runs/pretrain')
    p.add_argument('--steps',    type=int, default=100_000)
    p.add_argument('--bs',       type=int, default=64)
    p.add_argument('--lr',       type=float, default=3e-4)
    p.add_argument('--context_length', type=int, default=1000)
    p.add_argument('--d_model',  type=int, default=192)
    p.add_argument('--attn_pattern', default='local50,global100,local50,global100')
    p.add_argument('--mask_ratio', type=float, default=0.4)
    # visualisation
    p.add_argument('--viz_every', type=int, default=10,
                  help=">0 ⇒ dump a triptych every N evals")
    p.add_argument('--early_stop', type=int, default=30)
    p.add_argument('--log_every',  type=int, default=200)
    args = p.parse_args()

    # rotate previous run
    rd = Path(args.run_dir)
    if rd.exists():
        stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
        shutil.move(rd, Path('runs/archive')/f'{rd.name}_{stamp}')
    rd.mkdir(parents=True, exist_ok=True)

    Trainer(args).train()

if __name__ == '__main__':
    main()
