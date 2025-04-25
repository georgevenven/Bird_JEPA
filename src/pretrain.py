# ──────────────────────────────────────────────────────────────────────────────
# src/pretrain.py      • BirdJEPA encoder self‑supervised training (rect‑mask)
# ──────────────────────────────────────────────────────────────────────────────
import argparse, json, time, os, shutil, uuid, datetime as dt, math
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.multiprocessing
import torch.nn.functional as tF        # avoid name clash
import random, copy

from models          import BJConfig, Predictor
from utils           import load_pretrained_encoder
from data.bird_datasets import TorchSpecDataset
from sklearn.metrics import roc_auc_score          # future‑proof (unused now)

# ── logging helpers (must appear before Trainer uses them) ─────────────
class Tee:
    def __init__(self, fn: Path):
        fn.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(fn, "a", buffering=1)
    def __call__(self,*msg):
        txt = " ".join(str(m) for m in msg)
        print(txt); self.f.write(txt+"\n")
    def close(self): self.f.close()

@torch.no_grad()
def grad_norm(model: nn.Module)->float:
    g2=0.0
    for p in model.parameters():
        if p.grad is not None:
            g2+=p.grad.float().norm()**2
    return math.sqrt(g2)

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
        cfg   = BJConfig(d_model=args.d_model, n_mels=args.n_mels,
                         pattern=args.attn_pattern)
        self.enc = load_pretrained_encoder(cfg, None)          # start from scratch
        self.pred = Predictor(cfg.d_model)
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enc.to(self.dev); self.pred.to(self.dev)

        # belt-and-suspenders: always provide inference_forward
        if not hasattr(self.enc, "inference_forward"):
            self.enc.inference_forward = lambda x: (self.enc(x), None)

        # ── opt / sched ───────────────────────────────────────
        self.warmup_steps = 2000
        self.enc_frozen   = True
        for p in self.enc.parameters():
            p.requires_grad = False

        enc_lr  = args.lr
        pred_lr = args.lr * 3
        self.opt = torch.optim.AdamW(
            [{"params": self.enc.parameters(),  "lr": enc_lr,  "weight_decay":1e-2},
             {"params": self.pred.parameters(), "lr": pred_lr, "weight_decay":5e-3}]
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.opt, T_max=max(1,args.steps-self.warmup_steps),
                        eta_min=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=AMP)
        self.crit   = nn.MSELoss()

        # ── run dir / logging ──────────────────────────────────
        self.run_dir = Path(args.run_dir); self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir/'weights').mkdir(exist_ok=True)
        (self.run_dir/'imgs').mkdir(exist_ok=True)
        self.log = Tee(self.run_dir/'train_log.txt')

        # track
        self.best_val = 9e9; self.bad_evals = 0
        self.hist = {'step':[], 'train':[], 'val':[], 'grad':[]}
        self.alpha = 0.1

        # init EMA placeholders so .train() can update them
        self.train_ema = None
        self.val_ema   = None

        self._eval_counter = 0

        # ── EMA teacher (frozen) ───────────────────────────────
        self.teacher = copy.deepcopy(self.enc).eval()
        for p in self.teacher.parameters(): p.requires_grad = False
        self.mu_base   = 0.995
        self.mu_final  = 1.0
        self.mu_ramp   = 30_000            # linear ramp steps

    # -----------------------------------------------------------------
    # ──────────────────────────────────────────────────────────────
    #  rectangle mask on raw (F,T) spectrogram → pooled to tokens
    # ──────────────────────────────────────────────────────────────
    def _rect_mask(self, F_bins, T_bins, mask_p=.30, device="cpu"):
        target = int(mask_p * F_bins * T_bins)
        remain = target
        m = torch.zeros(F_bins, T_bins, dtype=torch.bool, device=device)

        while remain > 0:
            # choose height no bigger than what's left vertically
            h = random.randint(1, min(F_bins, remain))
            # cap width by leftover area / chosen height
            max_w = min(T_bins, max(1, remain // h))
            w = random.randint(1, max_w)

            # final safety: ensure area ≤ remain
            area = min(h * w, remain)
            # if area shrank, reduce width accordingly
            w = max(1, area // h)

            f0 = random.randint(0, F_bins - h)
            t0 = random.randint(0, T_bins - w)

            m[f0:f0 + h, t0:t0 + w] = True
            remain -= h * w

        return m

    # -----------------------------------------------------------------
    def _step(self, spec):
        spec = spec.to(self.dev)                  # (B,F,T)
        B,F_bins,T_bins = spec.shape

        # -------------- grid-aware mask on token grid -------------------
        # get token grid dims from encoder stem
        with torch.no_grad():
            dummy = torch.zeros(1, 1, F_bins, T_bins, device=spec.device)
            _, C, Fp, Tp = self.enc.stem(dummy).shape
        mask_tok = self._rect_mask(Fp, Tp, self.args.mask_ratio, device=spec.device).flatten()  # (Fp*Tp,)
        mask_tok = mask_tok.unsqueeze(0).repeat(B,1)                                            # (B, Fp*Tp)

        # ------- zero entire patches in spec BEFORE the stem ----------
        spec_masked = spec.clone()
        f_stride = F_bins // Fp
        t_stride = T_bins // Tp
        for f in range(Fp):
            for t in range(Tp):
                if mask_tok[0, f*Tp + t]:
                    f0, f1 = f*f_stride, (f+1)*f_stride
                    t0, t1 = t*t_stride, (t+1)*t_stride
                    spec_masked[:, f0:f1, t0:t1] = 0.0

        with torch.cuda.amp.autocast(enabled=AMP):
            spec_masked = spec_masked.unsqueeze(1)
            if hasattr(self.enc, "inference_forward"):
                z, _ = self.enc.inference_forward(spec_masked)
            else:
                z = self.enc(spec_masked)
            B, T_tok, D = z.shape

            # -------- teacher targets (full spec, no masking) ------------
            with torch.no_grad():
                full = spec.unsqueeze(1)
                z_tgt, _ = self.teacher.inference_forward(full) \
                           if hasattr(self.teacher, "inference_forward") \
                           else self.teacher(full)

            # -------- student prediction ----------------------------------
            z_ctx = z.clone(); z_ctx[mask_tok] = 0
            pred = self.pred(z_ctx, (Fp, Tp))
            loss = self.crit(pred[mask_tok], z_tgt[mask_tok])

        self.opt.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(list(self.enc.parameters())+
                                       list(self.pred.parameters()), 1.0)
        self.scaler.step(self.opt); self.scaler.update()

        # ── EMA update ----------------------------------------------------
        mu = self.mu_final if self.step >= self.mu_ramp else \
             self.mu_final - (self.mu_final - self.mu_base) * \
             (1 - self.step / self.mu_ramp)
        with torch.no_grad():
            for pt, ps in zip(self.teacher.parameters(), self.enc.parameters()):
                pt.mul_(mu).add_(ps.data, alpha=1-mu)

        # ── optional viz ───────────────────────────────────
        if self.args.eval_interval and (self.args.eval_interval > 0) and (hasattr(self, 'step') and (self.step % self.args.eval_interval == 0)):
            with torch.no_grad():
                s0 = spec[0].cpu()              # (F,T)
                # for viz, reconstruct a (F,T) mask from token mask
                m0 = torch.zeros(F_bins, T_bins, dtype=torch.bool)
                for f in range(Fp):
                    for t in range(Tp):
                        if mask_tok[0, f*Tp + t]:
                            f0, f1 = f*f_stride, (f+1)*f_stride
                            t0, t1 = t*t_stride, (t+1)*t_stride
                            m0[f0:f1, t0:t1] = True
                p0 = pred[0].cpu()              # (T,D)
                z0 = z[0].cpu()                 # (T,D)
            _viz_triptych(self.run_dir/'imgs', self.step, s0, m0, p0, z_tgt[0].cpu())
        return float(loss.item()), mask_tok

    # -----------------------------------------------------------------
    @torch.no_grad()
    def _eval(self):
        self.enc.eval(); self.pred.eval()
        tot = 0.0; n = 0
        for spec, *_ in self.val_dl:
            spec = spec.to(self.dev)                       # (B,F,T)
            B,F_bins,T_bins = spec.shape

            base = self._rect_mask(F_bins, T_bins, self.args.mask_ratio,
                                   device=spec.device)    # (F,T)
            mask3 = base.unsqueeze(0).repeat(B,1,1)        # (B,F,T)

            with torch.cuda.amp.autocast(enabled=AMP):
                s = spec.unsqueeze(1)
                z, _ = self.enc.inference_forward(s) if hasattr(self.enc,"inference_forward") \
                          else (self.enc(s), None)        # (B,T_tok,D)

                # pool mask to encoder grid (match stem output)
                _, C, Fp, Tp = self.enc.stem(s).shape   # <- 8,16 here
                k_f, k_t = F_bins // Fp, T_bins // Tp                # both ints
                pooled = F.max_pool2d(mask3.float().unsqueeze(1),
                                     kernel_size=(k_f, k_t),
                                     stride=(k_f, k_t))
                mask_tok = pooled.flatten(2).squeeze(1).bool()       # (B, Fp*Tp) == (B,128)

                z_ctx = z.clone(); z_ctx[mask_tok] = 0
                pred = self.pred(z_ctx, (Fp, Tp))
                tot += self.crit(pred[mask_tok], z[mask_tok]).item()*B
                n   += B

            if self.args.eval_interval and (self._eval_counter % self.args.eval_interval == 0):
                _viz_triptych(self.run_dir/'imgs', self._eval_counter,
                              spec[0].cpu(), mask3[0].cpu(),
                              pred[0].cpu(), z[0].cpu())
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
            tr_loss, m_tok = self._step(spec)
            if first_step:
                first_step = False            # scheduler already advanced in _step()
            else:
                if step > 1:           # skip the misleading first tick
                    self.sched.step()

            if step % self.args.eval_interval == 0:
                val_loss = self._eval()

                # EMA
                self.train_ema = tr_loss if self.train_ema is None else \
                                 self.alpha*tr_loss + (1-self.alpha)*self.train_ema
                self.val_ema   = val_loss if self.val_ema is None else \
                                 self.alpha*val_loss + (1-self.alpha)*self.val_ema

                gnorm = grad_norm(self.enc)+grad_norm(self.pred)
                lrs   = " ".join(f"lr{i}:{pg['lr']:.2e}"
                                 for i,pg in enumerate(self.opt.param_groups))
                self.log(f"step {step:07d} | train {self.train_ema:.4f} "
                         f"| val {self.val_ema:.4f} | gnorm {gnorm:.3e} | {lrs}")

                self.hist['step'].append(step)
                self.hist['train'].append(self.train_ema)
                self.hist['val'].append(self.val_ema)
                self.hist['grad'].append(gnorm)

                if self.val_ema < self.best_val - 1e-4:
                    self.best_val = self.val_ema; self.bad_evals = 0
                    torch.save({'enc': self.enc.state_dict(),
                                'pred': self.pred.state_dict()},
                               self.run_dir/'weights/best.pt')
                else:
                    self.bad_evals += 1
                if self.bad_evals >= self.args.early_stop:
                    print("early stop")
                    break

            # ---- staged unfreeze -------------------------------
            if self.enc_frozen and step >= self.warmup_steps:
                for p in self.enc.parameters():
                    p.requires_grad = True
                self.enc_frozen = False
                self.log(f"step {step}: encoder unfrozen")

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

    # ---------- infer token grid from pred shape ----------------------
    PF = 8                                   # freq remains fixed
    PT = pred.shape[0] // PF                 # time bins = tokens / 8

    # 1) error heat-map ------------------------------------------
    err = (pred - target).abs().mean(-1).cpu().numpy()        # (tokens,)
    err_grid = err.reshape(PF, PT)
    err_img = tF.interpolate(torch.from_numpy(err_grid)[None,None],
                             size=spec_ds.shape, mode="nearest"
               ).squeeze().numpy()

    # 2) teacher / student token norms ---------------------------
    teach = target.norm(dim=-1).cpu().numpy().reshape(PF,PT)
    stud  =  pred .norm(dim=-1).cpu().numpy().reshape(PF,PT)
    teach = (teach - teach.min()) / (teach.ptp()+1e-6)
    stud  = (stud  - stud .min()) / (stud .ptp()+1e-6)

    # 3) grid-downsampled input ----------------------------------
    spec_grid = tF.adaptive_avg_pool2d(
                    spec_ds.unsqueeze(0).unsqueeze(0), (PF,PT)
                ).squeeze().numpy()

    fig,ax = plt.subplots(2,3,figsize=(9,6),dpi=120)
    kw = dict(aspect='auto',origin='lower',cmap='magma')
    # top row ------------------------------------------------------
    ax[0,0].imshow(spec, **kw);           ax[0,0].set_title("input")
    ax[0,1].imshow(spec_masked, **kw);    ax[0,1].set_title("masked")
    ax[0,2].imshow(err_img, cmap='coolwarm', aspect='auto',
                   origin='lower', vmin=0, vmax=err_img.max());
    ax[0,2].set_title("error")
    # bottom row ---------------------------------------------------
    ax[1,0].imshow(spec_grid, **kw);      ax[1,0].set_title("grid")
    ax[1,1].imshow(stud.T,  cmap='viridis');ax[1,1].set_title("student |z|")
    ax[1,2].imshow(teach.T, cmap='viridis');ax[1,2].set_title("teacher |z|")

    for a in ax.ravel(): a.axis('off')
    out = Path(run_dir) / f"viz_{step:07d}.png"
    fig.tight_layout(); fig.savefig(out); plt.close(fig)

# ╭───────────────────────────────────────────────────────────────╮
# │ CLI                                                          │
# ╰───────────────────────────────────────────────────────────────╯
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', required=True)
    p.add_argument('--val_dir')
    p.add_argument('--run_dir',  default='runs/pretrain')
    p.add_argument('--steps',    type=int, default=300000)
    p.add_argument('--bs',       type=int, default=64)
    p.add_argument('--lr',       type=float, default=3e-4)
    p.add_argument('--context_length', type=int, default=256)
    p.add_argument('--d_model',  type=int, default=192)
    p.add_argument('--attn_pattern', default='local50,global100,local50,global100')
    p.add_argument('--mask_ratio', type=float, default=0.4)
    p.add_argument('--n_mels', type=int, default=128,
                   help='input spectrogram height (mel bins)')
    # visualisation
    p.add_argument('--eval_interval', type=int, default=200,
                  help="steps between val-loss / console log / viz dump")
    p.add_argument('--early_stop', type=int, default=30)
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
