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

# local imports (unchanged public interface)
from models  import BJConfig                            # encoder
from models.birdjepa import BirdJEPA
from utils   import load_pretrained_encoder                       # simple helper
from data.bird_datasets import BirdSpectrogramDataset
# ╭──────────────────────────────────────────────────────────────────────────╮
# │  Kaggle‑style macro ROC‑AUC – self‑contained, no external dependency     │
# ╰──────────────────────────────────────────────────────────────────────────╯
from sklearn.metrics import roc_auc_score
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
    g2=0.0
    for p in model.parameters():
        if p.grad is not None:
            g2 += p.grad.norm()**2
    return float(g2**0.5)

# ╭──────────────────────────────────────────────────────────────────────────╮
# │  classifier head                                                        │
# ╰──────────────────────────────────────────────────────────────────────────╯
class BetterHead(nn.Module):
    def __init__(self, d_model, n_cls, hid=256):
        super().__init__()
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1))
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Dropout(0.2),
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
        spec = spec.unsqueeze(1).transpose(2,3)                   # (B,1,T,F)
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
        self.args = args
        # ── data ───────────────────────────────────────────────
        self.train_ds = BirdSpectrogramDataset(args.train_spec_dir,
                                               segment_len=args.context_length,
                                               csv_path=args.train_csv)
        self.val_ds   = BirdSpectrogramDataset(args.val_spec_dir,
                                               segment_len=args.context_length,
                                               infinite=False,
                                               csv_path=args.train_csv)
        self.classes  = self.train_ds.classes
        self.train_dl = torch.utils.data.DataLoader(self.train_ds,
                                                    args.batch_size, shuffle=True,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True, drop_last=True)
        self.val_dl   = torch.utils.data.DataLoader(self.val_ds, 1,
                                                    shuffle=False, num_workers=0)

        # ── model ──────────────────────────────────────────────
        cfg = BJConfig(d_model=args.enc_width, pattern=args.attn_pattern)
        self.net = Net(args.pretrained_model_path, cfg,
                       len(self.classes))
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.dev)

        # freeze encoder if requested
        if args.freeze_encoder:
            for p in self.net.encoder.parameters(): p.requires_grad=False

        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.learning_rate)
        self.crit = nn.BCEWithLogitsLoss()

        # ── dirs / logging ─────────────────────────────────────
        self.run_dir = Path(args.output_dir); self.run_dir.mkdir(parents=True,exist_ok=True)
        (self.run_dir/"weights").mkdir(exist_ok=True)
        (self.run_dir/"config.json").write_text(json.dumps(vars(args),indent=2))
        self.log = Tee(self.run_dir/"train_log.txt")

        # state
        self.best_val = 9e9; self.bad_evals = 0

        # stats history
        self.hist = {"step":[], "train":[], "val":[], "grad":[]}

    # ----------------------------------------------------------
    def _label_tensor(self, fnames):
        y = torch.zeros(len(fnames), len(self.classes), device=self.dev)
        for i,f in enumerate(fnames):
            idx = self.train_ds.label_idx(f)
            if idx>=0: y[i,idx]=1
        return y

    # ----------------------------------------------------------
    def step(self, batch):
        spec, _, fn = batch                     # dataset returns (spec, lab, fn)
        spec = spec.to(self.dev)
        y    = self._label_tensor(fn)
        self.opt.zero_grad()
        loss = self.crit(self.net(spec), y)
        loss.backward(); self.opt.step()
        return float(loss.item())

    # ----------------------------------------------------------
    @torch.no_grad()
    def eval_loss(self, n=64):
        self.net.eval(); tot=0; cnt=0
        for spec,_,fn in self.val_dl:
            spec=spec.to(self.dev)
            y=self._label_tensor(fn)
            tot += self.crit(self.net(spec),y).item(); cnt+=1
            if cnt>=n: break
        self.net.train()
        return tot/cnt

    # ----------------------------------------------------------
    def train(self):
        ema_train=None; ema_val=None; α=.1
        step = 0
        while step < self.args.max_steps and step < 100:
            for batch in self.train_dl:
                step += 1
                l = self.step(batch)
                ema_train = l if ema_train is None else α*l+(1-α)*ema_train

                if step%self.args.eval_interval: continue
                v = self.eval_loss()
                ema_val = v if ema_val is None else α*v+(1-α)*ema_val

                gnorm = grad_norm(self.net)
                self.log(f"step {step:06d} | loss {ema_train:.4f} | "
                         f"val {ema_val:.4f} | gnorm {gnorm:.2f}")

                # record
                self.hist["step"].append(step)
                self.hist["train"].append(ema_train)
                self.hist["val"].append(ema_val)
                self.hist["grad"].append(gnorm)

                # early‑stop bookkeeping
                improved = v < self.best_val - 1e-5
                if improved:
                    self.best_val = v; self.bad_evals=0
                    self._save(step,'best')
                else:
                    self.bad_evals += 1

                if step%self.args.save_interval==0: self._save(step,'ckpt')
                if self.bad_evals>=self.args.early_stopping_patience:
                    self.log(f"early stop (no improve for {self.bad_evals} evals)")
                    break
                if step>=self.args.max_steps: break
            if step>=self.args.max_steps or self.bad_evals>=self.args.early_stopping_patience:
                break

        # --- after training loop: dump metrics and plots ---
        df = pd.DataFrame(self.hist)
        df.to_csv(self.run_dir/'metrics.csv', index=False)

        plt.figure(figsize=(5,4))
        plt.plot(df.step, df.train, label='train ema')
        plt.plot(df.step, df.val,   label='val ema')
        plt.xlabel('step'); plt.ylabel('loss'); plt.legend()
        plt.tight_layout(); plt.savefig(self.run_dir/'loss_curve.png', dpi=150); plt.close()

        plt.figure(figsize=(5,4))
        plt.plot(df.step, df.grad); plt.xlabel('step'); plt.ylabel('grad‑norm')
        plt.tight_layout(); plt.savefig(self.run_dir/'grad_norm.png', dpi=150); plt.close()

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

        ckpt = sorted((Path(args.output_dir)/"weights").glob("best_*.pt"))[-1]
        state=torch.load(ckpt,map_location="cpu")

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
                              opset_version=13)
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
        rows=[]; y_true=[]
        times=[]
        profile_path = None
        # drop explicit start; ORT starts auto‑profiling when enabled
        for i, (spec, _, fn) in enumerate(self.ds):
            # if i >= 100:
            #     break
            t0=time.time()
            try:
                segs = self._segment(spec)
                probs = [self._predict(s) for s in segs]

                base = re.sub(r'_segment_\d+$', '', Path(fn).stem)
                lab = np.zeros(len(self.classes)); lab[self.ds.label_idx(fn)] = 1
                for i, p in enumerate(probs):
                    row_id = f"{base}_{(i+1)*5}"
                    rows.append({"row_id": row_id, **{c: float(p[j]) for j, c in enumerate(self.classes)}})
                    y_true.append(lab)
            except (zipfile.BadZipFile, ValueError, OSError):
                self.log(f"skip corrupt {fn}")
                continue
            times.append(time.time()-t0)
        profile_path = self.sess.end_profiling()   # returns file path

        out_csv=Path(self.args.log_dir or self.args.output_dir)/self.args.submission_csv
        sub=pd.DataFrame(rows); sub.to_csv(out_csv,index=False,float_format="%.8f")
        sol=pd.DataFrame(y_true,columns=self.classes); sol.insert(0,"row_id",sub["row_id"])
        auc=kaggle_roc_auc(sol,sub)
        print(f"wrote {out_csv} | ROC‑AUC {auc:.4f}")
        if profile_path:
            self.log(f"onnx profile saved → {profile_path}")

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
    p.add_argument("--pretrained_model_path")
    p.add_argument("--onnx_model_path")
    p.add_argument("--log_dir")
    p.add_argument("--submission_csv", default="submission.csv")
    p.add_argument("--context_length",type=int,default=1000)
    p.add_argument("--batch_size",type=int,default=1)
    p.add_argument("--learning_rate",type=float,default=5e-4)
    p.add_argument("--max_steps",type=int,default=100000)
    p.add_argument("--eval_interval",type=int,default=100)
    p.add_argument("--save_interval",type=int,default=1000)
    p.add_argument("--early_stopping_patience",type=int,default=100)
    p.add_argument("--freeze_encoder",action="store_true")
    p.add_argument("--num_workers",type=int,default=4)
    p.add_argument("--enc_width",type=int,default=96,help="JEPA hidden size d_model")
    p.add_argument("--attn_pattern", default="local50,global100,local50,global100")
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