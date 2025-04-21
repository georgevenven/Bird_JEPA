# ──────────────────────────────────────────────────────────────────────────────
# src/finetune.py
# JEPA → classifier fine‑tuning   *and*   inference / submission builder
# ──────────────────────────────────────────────────────────────────────────────
import argparse, json, re, time
from pathlib import Path
import shutil, uuid, datetime as dt
import zipfile

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
import onnx, onnxruntime as ort
import os
import torch.multiprocessing

# local imports (unchanged public interface)
from models  import BJConfig                            # encoder
from models.birdjepa import BirdJEPA
from utils   import load_pretrained_encoder                       # simple helper
from data.bird_datasets import BirdSpectrogramDataset, TorchSpecDataset
# ╭──────────────────────────────────────────────────────────────────────────╮
# │  Kaggle‑style macro ROC‑AUC – self‑contained, no external dependency     │
# ╰──────────────────────────────────────────────────────────────────────────╯
from sklearn.metrics import roc_auc_score
# --- AMP setup ---
from contextlib import nullcontext
AMP = torch.cuda.is_available()
if AMP:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def kaggle_roc_auc(solution: pd.DataFrame, submission: pd.DataFrame,
                   row_id: str = "row_id") -> float:
    sol = solution.drop(columns=[row_id])
    sub = submission.drop(columns=[row_id])
    aucs=[]
    for c in sol.columns:
        if sol[c].sum()==0:                   # skip classes w/ zero positives
            continue
        try:
            aucs.append(roc_auc_score(sol[c].values, sub[c].values))
        except ValueError:                    # constant prediction
            pass
    return float(np.mean(aucs)) if aucs else np.nan

# ╭──────────────────────────────────────────────────────────────────────────╮
# │  misc helpers                                                           │
# ╰──────────────────────────────────────────────────────────────────────────╯
class Tee:
    """print(msg) *and* append to file."""
    def __init__(self, fn: Path):
        fn.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(fn, "a", buffering=1)
    def __call__(self, *msg):
        txt = " ".join(str(m) for m in msg)
        print(txt); self.f.write(txt+"\n")
    def close(self): self.f.close()

@torch.no_grad()
def grad_norm(model: nn.Module) -> float:
    g2 = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g2 += p.grad.float().norm()**2      # fp32
    return float(g2**0.5)

# ╭──────────────────────────────────────────────────────────────────────────╮
# │  classifier head                                                        │
# ╰──────────────────────────────────────────────────────────────────────────╯
class BetterHead(nn.Module):
    def __init__(self, d_model, n_cls, hid=512):
        super().__init__()
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1))
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Dropout(0.3),       # stronger regularisation
            nn.Linear(hid, n_cls))
    def forward(self, x):          # x (B,T,d)
        w = self.attn_pool(x)      # (B,T,1)
        x = (w * x).sum(1)         # weighted sum
        return self.mlp(x)

# ╭──────────────────────────────────────────────────────────────────────────╮
# │  Net = frozen (or not) encoder + head                                    │
# ╰──────────────────────────────────────────────────────────────────────────╯
class Net(nn.Module):
    def __init__(self, enc_ckpt:str|None, cfg:BJConfig,
                 n_cls:int):
        super().__init__()
        self.encoder = load_pretrained_encoder(cfg, enc_ckpt)     # freeze later
        self.clf     = BetterHead(cfg.d_model, n_cls)
    def forward(self, spec):                                      # (B,F,T)
        spec = spec.unsqueeze(1)                   # (B,1,F,T)
        if hasattr(self.encoder, "inference_forward"):
            emb, _ = self.encoder.inference_forward(spec)   # full BirdJEPA case
        else:
            emb = self.encoder(spec)                       # Sequential(stem,enc) case
        return self.clf(emb)

# ╭──────────────────────────────────────────────────────────────────────────╮
# │  TRAINER                                                                │
# ╰──────────────────────────────────────────────────────────────────────────╯
class Trainer:
    def __init__(self, args):
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.args = args
        # ── data ───────────────────────────────────────────────
        self.train_ds = TorchSpecDataset(args.train_spec_dir,
                                               segment_len=args.context_length,
                                               csv_path=args.train_csv)
        self.val_ds   = TorchSpecDataset(args.val_spec_dir,
                                               segment_len=args.context_length,
                                               infinite=False,
                                               csv_path=args.train_csv)
        self.classes  = self.train_ds.classes
        self.train_dl = torch.utils.data.DataLoader(self.train_ds,
                                                    batch_size=self.args.batch_size, shuffle=True,
                                                    num_workers=4,
                                                    pin_memory=True,
                                                    persistent_workers=True,
                                                    prefetch_factor=2,
                                                    drop_last=True)
        self.val_dl   = torch.utils.data.DataLoader(self.val_ds, 1,
                                                    shuffle=False, num_workers=0)

        # ── model ──────────────────────────────────────────────
        cfg = BJConfig(d_model=args.enc_width, pattern=args.attn_pattern)
        self.net = Net(args.pretrained_model_path, cfg,
                       len(self.classes))
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.dev)
        torch._dynamo.config.suppress_errors = True        # auto‑fallback to eager
        os.environ["TORCH_COMPILE_DISABLE"] = "1"

        # --- head-only warm-up ---
        self.warmup_steps = 2000
        self.enc_frozen = True
        for p in self.net.encoder.parameters():
            p.requires_grad = False

        enc_lr  = 1e-4
        head_lr = 3e-3
        self.opt = torch.optim.AdamW([
            {"params": self.net.encoder.parameters(), "lr": enc_lr, "weight_decay": 1e-2},
            {"params": self.net.clf.parameters(),     "lr": head_lr, "weight_decay": 5e-3},
        ])
        self.scaler = torch.cuda.amp.GradScaler(enabled=AMP)
        self.crit = nn.BCEWithLogitsLoss()

        # optional: cosine to 1e‑5
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=args.max_steps - self.warmup_steps, eta_min=1e-5)

        # ── dirs / logging ─────────────────────────────────────
        self.run_dir = Path(args.output_dir); self.run_dir.mkdir(parents=True,exist_ok=True)
        (self.run_dir/"weights").mkdir(exist_ok=True)
        (self.run_dir/"config.json").write_text(json.dumps(vars(args),indent=2))
        self.log = Tee(self.run_dir/"train_log.txt")

        # state
        self.best_val = 9e9; self.bad_evals = 0

        # stats history
        self.hist = {"step":[],            # global step
                     "train":[],           # EMA train‑loss
                     "val":[],             # EMA val‑loss
                     "auc":[],             # EMA val‑AUC
                     "grad":[]}            # grad‑norm

        self.alpha = 0.1              # EMA smoothing factor
        self.train_ema = None
        self.val_ema   = None

        # --- new for AUC EMA tracking ---
        self.ema_alpha = 0.1
        self.ema_auc   = None
        self.hist_auc  = []  # list of dicts for csv/plot

    # ----------------------------------------------------------
    def _label_tensor(self, fnames, eps: float = 0.05):  # label‑smoothing ε
        y = torch.zeros(len(fnames), len(self.classes), device=self.dev)
        for i, f in enumerate(fnames):
            idx = self.train_ds.label_idx(f)
            if idx >= 0:
                y[i, idx] = 1.0
        # smooth:   y ← (1‑ε)·y  +  ε/C
        y = y * (1 - eps) + eps / y.size(1)
        return y

    # ----------------------------------------------------------
    def step(self, batch):
        spec, _, fn = batch
        spec = spec.to(self.dev)
        y    = self._label_tensor(fn)

        with torch.cuda.amp.autocast():
            loss = self.crit(self.net(spec), y)

        self.opt.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)              # for true gnorm
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        return float(loss.item())

    # ----------------------------------------------------------
    def _validate(self, step, max_batches=64):
        """
        quick val: at most `max_batches` × batch_size samples
        returns scalar loss and macro ROC‑AUC (skips zero‑positive classes)
        """
        self.net.eval()
        tot_loss = 0.0; n_seen = 0
        y_all, p_all = [], []

        with torch.no_grad():
            for b, (spec, _, fnames) in enumerate(self.val_dl):
                spec = spec.to(self.dev, dtype=torch.float32, non_blocking=True)
                y    = self._label_tensor(fnames, eps=0.0)
                logit = self.net(spec)
                loss  = self.crit(logit, y)

                tot_loss += loss.item() * len(spec)
                n_seen   += len(spec)

                p_all.append(torch.sigmoid(logit).cpu())
                y_all.append(y.cpu())

                if b + 1 >= max_batches:
                    break

        val_loss = tot_loss / n_seen

        # --- macro ROC‑AUC (skip classes with no positives) -------------
        y_cat = torch.cat(y_all)
        p_cat = torch.cat(p_all)
        aucs  = []
        for c in range(y_cat.shape[1]):
            if y_cat[:, c].sum() == 0:
                continue
            try:
                aucs.append(roc_auc_score(y_cat[:, c], p_cat[:, c]))
            except ValueError:        # constant prediction
                pass
        val_auc = float(np.mean(aucs)) if aucs else float("nan")

        # print(f"[val] step {step:06d}  loss {val_loss:.4f}  AUC {val_auc:.4f}")
        pass
        self.net.train()
        return val_loss, val_auc

    # ----------------------------------------------------------
    def train(self):
        step      = 0
        dl_iter   = iter(self.train_dl)
        while step < self.args.max_steps:
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(self.train_dl)
                continue

            step += 1
            l = self.step(batch)                       # back‑prop
            if self.sched is not None:                 # only if scheduler
                self.sched.step()
            self.train_ema = l if self.train_ema is None else \
                             self.alpha*l + (1-self.alpha)*self.train_ema

            # ────────────────────────────────────────────────
            # validate + log **only every eval_interval steps**
            # ────────────────────────────────────────────────
            if step % self.args.eval_interval == 0:
                v_loss, v_auc = self._validate(step, max_batches=16)
                # --- EMAs ---------------------------------------------------
                self.val_ema = v_loss if self.val_ema is None else \
                               self.alpha*v_loss + (1-self.alpha)*self.val_ema
                self.auc_ema = v_auc  if getattr(self, 'auc_ema', None) is None else \
                               self.alpha*v_auc  + (1-self.alpha)*self.auc_ema

                gnorm = grad_norm(self.net)
                lrs   = [pg["lr"] for pg in self.opt.param_groups]
                lr_txt = " ".join(f"lr{i}:{lr:.2e}" for i, lr in enumerate(lrs))
                scale = self.scaler.get_scale()

                self.log(f"step {step:06d} | train {self.train_ema:.4f} | "
                         f"val {self.val_ema:.4f} | AUC {self.auc_ema:.3f} "
                         f"| gnorm {gnorm:.4e} | scale {scale:.1f} | {lr_txt}")

                # record history
                self.hist["step"].append(step)
                self.hist["train"].append(self.train_ema)
                self.hist["val"].append(self.val_ema)
                self.hist["auc"].append(self.auc_ema)
                self.hist["grad"].append(gnorm)

                # early‑stopping bookkeeping, ckpt save, etc.
                improved = v_loss < self.best_val - 1e-5
                if improved:
                    self.best_val = v_loss; self.bad_evals = 0
                    self._save(step, 'best')
                else:
                    self.bad_evals += 1

                if step % self.args.save_interval == 0:
                    self._save(step, 'ckpt')

                if self.bad_evals >= self.args.early_stopping_patience:
                    self.log(f"early stop (no improve for {self.bad_evals} evals)")
                    break

            if step>=self.args.max_steps: break

            # --------------------------------------------------
            # staged‑unfreeze: flip encoder gradients on *once*
            # --------------------------------------------------
            if self.enc_frozen and step >= self.warmup_steps:
                for p in self.net.encoder.parameters():
                    p.requires_grad = True
                self.enc_frozen = False
                self.log(f"step {step}: encoder unfrozen (lr={self.opt.param_groups[0]['lr']:.2e})")

        # --- after training loop: dump metrics and plots ---
        df = pd.DataFrame(self.hist)
        df.to_csv(self.run_dir/'metrics.csv', index=False)

        plt.figure(figsize=(5,4))
        plt.plot(df.step, df.train, label='train ema')
        plt.plot(df.step, df.val,   label='val ema')
        plt.xlabel('step'); plt.ylabel('loss'); plt.legend()
        plt.tight_layout(); plt.savefig(self.run_dir/'loss_curve.png', dpi=150); plt.close()

        # ---------- AUC curve -------------------------------------------
        if df.auc.notna().any():
            plt.figure(figsize=(5,4))
            plt.plot(df.step, df.auc)
            plt.xlabel('step'); plt.ylabel('val AUC (EMA)')
            plt.tight_layout(); plt.savefig(self.run_dir/'auc_curve.png', dpi=150); plt.close()

    # ----------------------------------------------------------
    def _save(self,step,tag):
        torch.save({'step':step,'model':self.net.state_dict()},
                   self.run_dir/'weights'/f'{tag}_{step}.pt')
        self.log(f"saved {tag}_{step}.pt")

# ╭──────────────────────────────────────────────────────────────────────────╮
# │  INFERENCE                                                              │
# ╰──────────────────────────────────────────────────────────────────────────╯
class Infer:
    def __init__(self,args):
        self.args=args
        self.ds  = BirdSpectrogramDataset(args.val_spec_dir,
                                          segment_len=None, infinite=False,
                                          csv_path=args.train_csv)
        self.classes=self.ds.classes

        ckpt = Path(args.ckpt) if getattr(args, "ckpt", None) else \
               sorted((Path(args.output_dir)/"weights").glob("best_*.pt"))[-1]
        state=torch.load(ckpt,map_location="cpu", weights_only=True)

        cfg=BJConfig(d_model=args.enc_width, pattern=args.attn_pattern)
        self.net=Net(args.pretrained_model_path,cfg,len(self.classes))
        self.net.load_state_dict(state["model"]); self.net.eval()

        # ---- ONNX export (once) ---------------------------------
        self.onnx_path = Path(args.onnx_model_path or args.output_dir)/"model.onnx"
        if not self.onnx_path.exists():
            dummy=torch.randn(1,513,args.context_length)
            torch.onnx.export(self.net,dummy,self.onnx_path,
                              input_names=['input'],output_names=['out'],
                              dynamic_axes={'input':{0:'b'},'out':{0:'b'}},
                              opset_version=18)
        opt = ort.SessionOptions()
        opt.enable_profiling = True
        self.sess=ort.InferenceSession(str(self.onnx_path),
                                       sess_options=opt,
                                       providers=['CPUExecutionProvider'])

        self.log=Tee(Path(args.log_dir or args.output_dir)/"infer_log.txt")

    # ----------------------------------------------------------
    def _segment(self,spec,seg=1000):
        if spec.shape[-1]%seg:
            spec=F.pad(spec,(0,seg-spec.shape[-1]%seg))
        return spec.unfold(-1,seg,seg).squeeze(0).permute(1,0,2)

    # ----------------------------------------------------------
    def _predict(self,spec):
        out=self.sess.run(None,{self.sess.get_inputs()[0].name:
                                spec.numpy().astype(np.float32)[None]})[0]
        return 1/(1+np.exp(-out)).squeeze(0)

    # ----------------------------------------------------------
    def run(self):
        if len(self.ds) == 0:
            print("empty spec dir – nothing to score"); return

        val_loader = torch.utils.data.DataLoader(
            self.ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        rows=[]; y_true=[]
        times=[]
        profile_path = None
        # drop explicit start; ORT starts auto‑profiling when enabled
        for i, (spec, _, fn) in enumerate(val_loader):
            t0=time.time()
            try:
                # ── hard‑trim to 60 s (12 000 frames) ────────────────
                spec = spec[..., :12_000]          # if already shorter, noop

                segs = self._segment(spec)         # now 1‑to‑12 chunks, never 13
                probs = [self._predict(s) for s in segs]

                base = re.sub(r'_segment_\d+$', '', Path(fn[0]).stem)
                lab = np.zeros(len(self.classes)); lab[self.ds.label_idx(fn[0])] = 1
                for i, p in enumerate(probs):
                    row_id = f"{base}_{(i+1)*5}"
                    rows.append({"row_id": row_id, **{c: float(p[j]) for j, c in enumerate(self.classes)}})
                    y_true.append(lab)
            except (zipfile.BadZipFile, ValueError, OSError):
                self.log(f"skip corrupt {fn}")
                continue
            times.append(time.time()-t0)
        profile_path = self.sess.end_profiling()   # returns file path

        # -----------------------------------------------------------
        # build dataframe with all classes, even if rows / some cols missing
        cols = ["row_id"] + list(self.classes)
        if rows:
            sub = pd.DataFrame(rows)
            sub = sub.reindex(columns=cols, fill_value=0.0)
        else:                                 # no clips found → header‑only file
            sub = pd.DataFrame(columns=cols)

        out_csv = Path(self.args.log_dir or self.args.output_dir) / self.args.submission_csv
        sub.to_csv(out_csv, index=False, float_format="%.8f")

        if y_true:                           # if we actually had ground‑truth
            sol = pd.DataFrame(y_true, columns=self.classes)
            sol.insert(0, "row_id", sub["row_id"])
            auc = kaggle_roc_auc(sol, sub)
            print(f"wrote {out_csv} | ROC‑AUC {auc:.4f}")
        else:
            print(f"wrote {out_csv} | no validation clips found – skipped AUC")

        # ── timing summary ─────────────────────────────────────
        if times:
            tot, avg = sum(times), np.mean(times)
            self.log(f"processed {len(times)} files | "
                     f"total {tot:.1f}s | mean {avg*1000:.1f} ms per file")

# ╭──────────────────────────────────────────────────────────────────────────╮
# │  CLI                                                                    │
# ╰──────────────────────────────────────────────────────────────────────────╯
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["train","infer"],required=True)
    p.add_argument("--train_spec_dir"); p.add_argument("--val_spec_dir")
    p.add_argument("--output_dir",default="runs/finetuned")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--pretrained_model_path", default="")
    p.add_argument("--onnx_model_path")
    p.add_argument("--log_dir")
    p.add_argument("--submission_csv", default="submission.csv")
    p.add_argument("--context_length",type=int,default=1000)
    p.add_argument("--batch_size",type=int,default=320)
    p.add_argument("--learning_rate",type=float,default=5e-3)
    p.add_argument("--max_steps",type=int,default=100000)
    p.add_argument("--eval_interval",type=int,default=100)
    p.add_argument("--save_interval",type=int,default=1000)
    p.add_argument("--early_stopping_patience", default=300, type=int)
    p.add_argument("--freeze_encoder",action="store_true")
    p.add_argument("--num_workers",type=int,default=4)
    p.add_argument("--enc_width",type=int,default=192,help="JEPA hidden size d_model")
    p.add_argument("--attn_pattern", default="local50,global100,local50,global100")
    p.add_argument("--ckpt", help="explicit path to .pt checkpoint")
    args=p.parse_args()

    run_root = Path(args.output_dir)
    if args.mode == "train" and run_root.exists():
        from shutil import move
        import uuid, datetime as dt
        stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        tag   = uuid.uuid4().hex[:6]
        arch  = Path('runs/archive')/f'{run_root.name}_{stamp}_{tag}'
        arch.parent.mkdir(parents=True, exist_ok=True)
        move(run_root, arch)
        print(f'[finetune] previous run moved → {arch}')
    run_root.mkdir(parents=True, exist_ok=True)

    if args.mode=="train":
        Trainer(args).train()
    else:
        Infer(args).run()

if __name__=="__main__":
    main()