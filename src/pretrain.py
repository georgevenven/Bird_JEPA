# ──────────────────────────────────────────────────────────────────────────────
# src/pretrain.py      • BirdJEPA encoder self‑supervised training (rect‑mask)
# ──────────────────────────────────────────────────────────────────────────────
import argparse, json, time, os, shutil, uuid, datetime as dt
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.multiprocessing

from models          import BJConfig
from utils           import load_pretrained_encoder
from data.bird_datasets import TorchSpecDataset

# ---------------------------------------------------------------
AMP = torch.cuda.is_available()
if AMP:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True

# ╭───────────────────────────────────────────────────────────────╮
# │ small predictor g(·) → d‑model                               │
# ╰───────────────────────────────────────────────────────────────╯
class Predictor(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 2*d),
            nn.GELU(),
            nn.Linear(2*d, d))
    def forward(self, x): return self.g(x)

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

    # -----------------------------------------------------------------
    def _mask(self, B, T, device):
        """
        Rectangle‑style masking on the token timeline (B,T).
        We create random ⌈mask_ratio·T⌉ token indices per sample.
        """
        n_mask = int(T * self.args.mask_ratio)
        idx = torch.rand(B, T, device=device).argsort(dim=-1)[:, :n_mask]
        m   = torch.zeros(B, T, dtype=torch.bool, device=device)
        m.scatter_(1, idx, True)
        return m                                  # (B,T) True = masked

    # -----------------------------------------------------------------
    def _step(self, spec):
        spec = spec.to(self.dev)
        with torch.cuda.amp.autocast(enabled=AMP):
            spec = spec.unsqueeze(1)                                # (B,1,F,T)
            if hasattr(self.enc, "inference_forward"):
                z, _ = self.enc.inference_forward(spec)             # full BirdJEPA
            else:
                z = self.enc(spec)                                  # plain Sequential
            B, T, D = z.shape
            mask = self._mask(B, T, z.device)                      # (B,T)

            z_ctx = z.clone()
            z_ctx[mask] = 0                                         # stop‑grad on masked
            pred = self.pred(z_ctx)                                 # (B,T,d)

            loss = self.crit(pred[mask], z[mask])

        self.opt.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(list(self.enc.parameters())+
                                       list(self.pred.parameters()), 1.0)
        self.scaler.step(self.opt); self.scaler.update()
        self.sched.step()
        return float(loss.item())

    # -----------------------------------------------------------------
    @torch.no_grad()
    def _eval(self):
        self.enc.eval(); self.pred.eval()
        tot = 0.0; n = 0
        for spec, *_ in self.val_dl:
            spec = spec.to(self.dev)
            with torch.cuda.amp.autocast(enabled=AMP):
                z, _ = self.enc.inference_forward(spec.unsqueeze(1))
                B,T,D = z.shape
                mask = self._mask(B,T,z.device)
                z_ctx = z.clone(); z_ctx[mask] = 0
                pred = self.pred(z_ctx)
                tot += self.crit(pred[mask], z[mask]).item()*B
                n   += B
        self.enc.train(); self.pred.train()
        return tot/n

    # -----------------------------------------------------------------
    def train(self):
        dl_iter = iter(self.train_dl); step = 0
        while step < self.args.steps:
            try:       spec, *_ = next(dl_iter)
            except StopIteration:
                dl_iter = iter(self.train_dl); continue

            step += 1
            tr_loss = self._step(spec)

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
