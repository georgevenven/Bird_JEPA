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
import warnings # For eval shape mismatch warning

from models          import BJConfig, Predictor, DecoderPredictor
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

@torch.no_grad()
def param_norm(model: nn.Module) -> float:
    """Computes the overall L2 norm of all parameters in a model."""
    # Ensure calculations happen on the correct device
    # Use the device of the first parameter; assumes all params are on the same device
    device = next(model.parameters()).device
    total_norm_sq = torch.tensor(0.0, device=device)
    for p in model.parameters():
            total_norm_sq += p.norm(2).pow(2)
    return total_norm_sq.sqrt().item()

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
                                    infinite=True)
        self.train_dl = torch.utils.data.DataLoader(self.ds,  batch_size=args.bs,
                                                    shuffle=True,  num_workers=4,
                                                    pin_memory=True,  drop_last=True)
        self.val_dl   = torch.utils.data.DataLoader(self.val, batch_size=args.bs,
                                                    shuffle=True, num_workers=4)

        # ── model  (encoder + predictor) ────────────────────────
        self.cfg = BJConfig(d_model=args.d_model, n_mels=args.n_mels,
                         pattern=args.attn_pattern)
        if not hasattr(self.cfg, 'pred_layers'):
            self.cfg.pred_layers = 4

        self.enc = load_pretrained_encoder(self.cfg, None)          # start from scratch
        self.pred = DecoderPredictor(self.cfg.d_model, self.cfg.n_heads, self.cfg.pred_layers, self.cfg.ff_mult)
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.enc.to(self.dev); self.pred.to(self.dev)

        # belt-and-suspenders: always provide inference_forward
        if not hasattr(self.enc, "inference_forward"):
            self.enc.inference_forward = lambda x: (self.enc(x), None)

        # cache token grid dims and strides for fast masking
        with torch.no_grad():
            d = torch.zeros(1,1,self.cfg.n_mels,args.context_length, device=self.dev)
            _,_,self.Fp,self.Tp = self.enc.stem(d).shape
        self.f_stride = self.cfg.n_mels        // self.Fp
        self.t_stride = args.context_length // self.Tp

        # ── opt / sched ───────────────────────────────────────
        enc_lr  = 1e-4  # lower encoder learning rate
        pred_lr = 3e-4 # lower predictor learning rate
        self.opt = torch.optim.AdamW(
            [{"params": self.enc.parameters(),  "lr": enc_lr,  "weight_decay": 0.0},
             {"params": self.pred.parameters(), "lr": pred_lr, "weight_decay": 0.0}]
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.opt, T_max=max(1,args.steps),
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
        self.mu_base   = 0.95
        self.mu_final  = 0.999
        self.mu_ramp   = 50000

        # --- Attach pre-loss normalization components to the predictor module ---
        self.pred.pre_loss_ln = nn.LayerNorm(self.cfg.d_model, elementwise_affine=False).to(self.dev)
        # no running-mean centering this time – just the LN is enough

        # -------- resume logic ------------------------------------
        if args.resume:
            ckpts = sorted((self.run_dir/'weights').glob('step_*.pt'))
            # Initialize step to 0 in case no checkpoints are found
            self.step = 0
            if ckpts:
                last = ckpts[-1]
                print(f'[resume] loading {last}')
                pay  = torch.load(last, map_location='cpu')
                self.enc.load_state_dict(pay['enc'])
                self.pred.load_state_dict(pay['pred'])
                self.best_val   = pay.get('val_ema', self.best_val)
                # restore teacher from encoder weights
                self.teacher.load_state_dict(self.enc.state_dict())
                # recover step counter so lr-sched keeps in sync
                self.step = int(last.stem.split('_')[-1])
                for _ in range(self.step):        # fast-fwd scheduler
                    self.sched.step()
                print(f'[resume] starting from step {self.step}')
                self._warm_resumed = True          # flag

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

        # vectorized, gpu-native masking (no copies, no python triple-loop)
        pix_mask = torch.kron(
            mask_tok.view(B,Fp,Tp).float(),
            torch.ones((self.f_stride, self.t_stride), device=spec.device)
        ).bool()                                            # (B,F_bins,T_bins)
        spec_masked = spec.masked_fill(pix_mask, 0.)

        with torch.cuda.amp.autocast(enabled=AMP):
            spec_masked = spec_masked.unsqueeze(1)
            if hasattr(self.enc, "inference_forward"):
                z_student_masked_input, _ = self.enc.inference_forward(spec_masked)
            else:
                z_student_masked_input = self.enc(spec_masked)
            B, T_tok, D = z_student_masked_input.shape

            # -------- teacher targets (full spec, no masking) ------------
            with torch.no_grad():
                full = spec.unsqueeze(1)
                z_tgt, _ = self.teacher.inference_forward(full) \
                           if hasattr(self.teacher, "inference_forward") \
                           else self.teacher(full)

            # -------- student prediction ----------------------------------
            pred_masked_all = self.pred(z_student_masked_input, mask_tok, (Fp, Tp))  # (ΣM_i, D)
            sizes      = mask_tok.sum(1).tolist()            # list of M_i per sample
            pred_chunks= pred_masked_all.split(sizes, dim=0)
            pred_masked_0 = pred_chunks[0]                   # (M_0, D)
            token_mask_0  = mask_tok[0].cpu()

            # ----- LN-before-loss ----------------------------------------
            loss = self.crit(
                self.pred.pre_loss_ln(pred_masked_all),
                self.pred.pre_loss_ln(z_tgt[mask_tok])
            )

            # variance diagnostics
            var_pred    = pred_masked_all.var(dim=0).mean()
            var_teacher = z_tgt[mask_tok].var(dim=0).mean()
            var_student = z_student_masked_input.var(dim=0).mean()

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

        # ── optional viz (moved from _eval) ───────────────────────────────────
        if self.step % self.args.eval_interval == 0 and B > 0: # Ensure batch size > 0
            with torch.no_grad():
                # Reconstruct the pixel-level mask for the first item for visualization
                # This reflects the masking applied *before* the encoder's stem
                mask_viz_single = torch.zeros(F_bins, T_bins, dtype=torch.bool, device=spec.device)
                f_stride = F_bins // Fp
                t_stride = T_bins // Tp
                for f in range(Fp):
                    for t in range(Tp):
                        if mask_tok[0, f*Tp + t]: # Use the first item's token mask
                            f0, f1 = f*f_stride, (f+1)*f_stride
                            t0, t1 = t*t_stride, (t+1)*t_stride
                            mask_viz_single[f0:f1, t0:t1] = True

                # Call visualization function
                # Pass the student's actual output (from masked input) for visualization
                _viz_triptych(
                    self.run_dir/'imgs', f"step_{self.step:07d}",
                    spec[0].cpu(),
                    mask_viz_single.cpu(),
                    pred_masked_0.cpu(),          # ← only sample-0 preds
                    token_mask_0, (Fp, Tp),
                    z_tgt[0].cpu(),
                    z_student_masked_input[0].cpu()
                )

        # Return training loss and the token mask (mask_tok might be useful elsewhere)
        return float(loss.item()), mask_tok, (
            var_student.item(),
            var_teacher.item(),
            var_pred.item())

    # -----------------------------------------------------------------
    @torch.no_grad()
    def _eval(self):
        self.enc.eval(); self.pred.eval()
        with torch.no_grad():
            spec, *_ = next(iter(self.val_dl))   # just ONE batch
            spec = spec.to(self.dev)
            B,F_bins,T_bins = spec.shape
            if B == 0: return 0.0 # Handle empty batch case

            # --- Generate mask (similar logic to _step for consistency) ---
            # Get token grid dims
            # Need to do this on device
            dummy = torch.zeros(1, 1, F_bins, T_bins, device=self.dev)
            _, C, Fp, Tp = self.enc.stem(dummy).shape
            # Generate token mask
            mask_tok = self._rect_mask(Fp, Tp, self.args.mask_ratio, device=spec.device).flatten()
            mask_tok = mask_tok.unsqueeze(0).repeat(B,1) # (B, Fp*Tp)

            # vectorized, gpu-native masking (no copies, no python triple-loop)
            pix_mask = torch.kron(
                mask_tok.view(B,Fp,Tp).float(),
                torch.ones((self.f_stride, self.t_stride), device=spec.device)
            ).bool()                                            # (B,F_bins,T_bins)
            spec_masked = spec.masked_fill(pix_mask, 0.)

            with torch.cuda.amp.autocast(enabled=AMP):
                spec_masked_unsqueezed = spec_masked.unsqueeze(1)
                z_student_masked_input = self.enc(spec_masked_unsqueezed) # Student on masked input
                full = spec.unsqueeze(1)
                z_tgt = self.teacher(full) # Teacher on unmasked input
                pred_masked = self.pred(z_student_masked_input, mask_tok, (Fp, Tp))  # already flat
                tgt_masked  = z_tgt[mask_tok]                                        # flat too

                # Handle potential edge case where masking results in zero tokens selected in a small batch
                if pred_masked.shape[0] == 0 or tgt_masked.shape[0] == 0:
                    warnings.warn(f"Skipping eval batch due to zero masked tokens. Pred shape: {pred_masked.shape}, Target shape: {tgt_masked.shape}")
                    return 0.0 # Or np.nan, or some indicator of no loss computed
                assert pred_masked.shape == tgt_masked.shape, f"Shape mismatch in eval: Pred {pred_masked.shape}, Target {tgt_masked.shape}"
                assert pred_masked.shape[1] == self.cfg.d_model, "Dimension mismatch in eval pred"
                assert tgt_masked.shape[1] == self.cfg.d_model, "Dimension mismatch in eval target"

                # the same LN path as train (no EMA update)
                pred_norm = self.pred.pre_loss_ln(pred_masked)
                tgt_norm  = self.pred.pre_loss_ln(tgt_masked)
                loss      = self.crit(pred_norm, tgt_norm).item()
                with torch.no_grad():
                    cos = F.cosine_similarity(pred_masked, tgt_masked, dim=-1)
                    mean_cos = cos.mean().item()

        self.enc.train(); self.pred.train()
        return loss, mean_cos

    # -----------------------------------------------------------------
    def train(self):
        dl_iter = iter(self.train_dl); step = getattr(self, 'step', 0)   # maybe restored
        first_step = True
        while step < self.args.steps:
            try:       spec, *_ = next(dl_iter)
            except StopIteration:
                dl_iter = iter(self.train_dl); continue

            step += 1
            self.step = step
            tr_loss, m_tok, (v_s, v_t, v_p) = self._step(spec)
            if first_step:
                first_step = False            # scheduler already advanced in _step()
            else:
                if hasattr(self, '_warm_resumed') and self._warm_resumed:
                    self._warm_resumed = False     # skip sched.step() this very first loop
                elif step > 1:           # skip the misleading first tick
                    self.sched.step()

            if step % self.args.eval_interval == 0:
                val_loss, val_cos = self._eval()           # still compute it

                # EMA
                self.train_ema = tr_loss if self.train_ema is None else \
                                 self.alpha*tr_loss + (1-self.alpha)*self.train_ema
                self.val_ema   = val_loss if self.val_ema is None else \
                                 self.alpha*val_loss + (1-self.alpha)*self.val_ema

                gnorm = grad_norm(self.enc) + grad_norm(self.pred)
                lrs   = " ".join(f"lr{i}:{pg['lr']:.2e}"
                                 for i,pg in enumerate(self.opt.param_groups))

                diag  = (f" | var_s {v_s:.3e}"
                         f" var_t {v_t:.3e}"
                         f" var_p {v_p:.3e}")

                msg = (f"step {step:07d} | train_raw {tr_loss:.4f} | val_raw {val_loss:.4f} "
                       f"| train_ema {self.train_ema:.4f} | val_ema {self.val_ema:.4f}"
                       f"{diag} | gnorm {gnorm:.2e} | {lrs} | mean_cos {val_cos:.3f}")
                self.log(msg)

                # Track history
                self.hist['step'].append(step)
                self.hist['train'].append(tr_loss)
                self.hist['val'].append(val_loss)

                ckpt_path = self.run_dir/'weights'/f'step_{step:07d}.pt'
                payload = {'enc': self.enc.state_dict(),
                           'pred': self.pred.state_dict(),
                           'val_ema': self.val_ema}
                torch.save(payload, ckpt_path)                     # always keep per-step

                if self.val_ema < self.best_val - 1e-4:            # overwrite best
                    self.best_val = self.val_ema; self.bad_evals = 0
                    torch.save(payload, self.run_dir/'weights'/'best.pt')
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
# global cache for a 3-vector PCA basis
_PCA_BASIS = None   # will hold a (D,3) tensor on first call
def _viz_triptych(run_dir, step,
                  spec, pix_mask,
                  pred_masked, token_mask, grid_shape,
                  target, stud):
    """
    spec         : (F,T)         cpu
    pix_mask     : (F,T) bool    pixel-level for first row
    pred_masked  : (M,D)         masked-token preds  (M = token_mask.sum())
    token_mask   : (T_all,) bool flat token mask
    grid_shape   : (PF,PT)
    target, stud : (T_all,D)
    """
    import matplotlib.pyplot as plt, torch.nn.functional as tF, numpy as np, torch

    PF, PT = grid_shape
    F_bins, T_full = spec.shape
    device = pred_masked.device

    # ---------- build full-grid tensors ---------------------------------
    D = target.shape[-1]
    pred_full  = torch.zeros(PF*PT, D, device=device)       # zeros = black
    err_full   = torch.zeros(PF*PT,     device=device)

    masked_idx = torch.nonzero(token_mask, as_tuple=False).squeeze(-1)
    pred_full[masked_idx] = pred_masked                      # scatter
    err_full[masked_idx]  = (pred_masked -
                             target[masked_idx]).pow(2).mean(-1)

    pred_full  = pred_full.view(PF, PT, D)
    err_full   = err_full.view(PF, PT)

    # -------- PCA->RGB with shared basis -------------------------------------
    def _pca_img_shared(z, fit=False):
        # z : (T,D) torch cpu
        global _PCA_BASIS
        x = z.float().view(PF*PT, -1)          # (tokens,D)

        # STEP 1 – fit basis once on teacher (now: only on masked rows)
        if fit or _PCA_BASIS is None:
            # Only use masked rows for PCA fit
            masked_rows = torch.nonzero(token_mask, as_tuple=False).squeeze(-1)
            x_masked = x[masked_rows]
            mu = x_masked.mean(0, keepdims=True)
            _, _, vh = torch.linalg.svd((x_masked - mu), full_matrices=False)
            _PCA_BASIS = vh[:3].T              # (D,3)

        # STEP 2 – project any x onto cached basis
        mu = x.mean(0, keepdims=True)
        y = (x - mu) @ _PCA_BASIS             # (tokens,3)
        pc = y.view(PF, PT, 3)
        pc -= pc.amin(dim=(0,1), keepdim=True)
        pc /= pc.amax(dim=(0,1), keepdim=True) + 1e-6
        return pc

    teach_img = _pca_img_shared(target, fit=True)     # fit basis
    stud_img  = _pca_img_shared(stud)                 # project
    pred_img  = _pca_img_shared(pred_full)            # project

    # ---------- prep spectrogram panels ---------------------------------
    if T_full != pix_mask.shape[1]:
        spec_ds = tF.interpolate(spec[None,None], size=pix_mask.shape[1],
                                 mode="nearest").squeeze()
    else:
        spec_ds = spec
    spec_masked = spec_ds.clone(); spec_masked[pix_mask] = 0.0

    # ---------- plot ----------------------------------------------------
    fig, ax = plt.subplots(2,3, figsize=(12,4), dpi=120)
    kw  = dict(aspect='auto', origin='lower', cmap='magma')
    kw2 = dict(aspect='auto', origin='lower', interpolation='nearest')

    ax[0,0].imshow(spec,       **kw);  ax[0,0].set_title('input')
    ax[0,1].imshow(spec_masked,**kw);  ax[0,1].set_title('masked')
    ax[0,2].imshow(err_full.cpu(), cmap='coolwarm', **kw2); ax[0,2].set_title('MSE')

    ax[1,0].imshow(teach_img, **kw2); ax[1,0].set_title('teacher')
    ax[1,1].imshow(stud_img,  **kw2); ax[1,1].set_title('student')
    ax[1,2].imshow(pred_img,  **kw2); ax[1,2].set_title('pred')

    for a in ax.ravel(): a.axis('off')
    fig.tight_layout()
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(run_dir)/f"viz_{step}.png")
    plt.close(fig)

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
    p.add_argument('--resume', action='store_true',
                   help='continue training from runs/<run_dir> if it already exists')
    args = p.parse_args()

    # rotate previous run
    rd = Path(args.run_dir)
    if rd.exists() and not args.resume:
        stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
        shutil.move(rd, Path('runs/archive')/f'{rd.name}_{stamp}')
        rd.mkdir(parents=True, exist_ok=True)
    elif not rd.exists():
        rd.mkdir(parents=True, exist_ok=True)

    Trainer(args).train()

if __name__ == '__main__':
    main()