# src/finetuning.py
# end‑to‑end training + inference script with on‑the‑fly kaggle ROC‑AUC

import argparse, json, shutil, time, re
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
import onnx, onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from datamodule import BirdDataModule
from model import BirdJEPA           # unchanged upstream import
from utils import load_model         # unchanged upstream import
from collate import masking_collate  # reuse

# ------------------------------------------------------------------
# self‑contained kaggle‑style macro ROC‑AUC (no external deps)
from sklearn.metrics import roc_auc_score

def kaggle_roc_auc(solution_df: pd.DataFrame,
                   submission_df: pd.DataFrame,
                   row_id: str = "row_id") -> float:
    """
    Replicates Kaggle's BirdCLEF macro‑ROC‑AUC metric:
      • drop the row‑id col
      • ignore any species with zero positives in the ground‑truth
      • average AUC over the remaining species
      • silently skip constant‑prediction errors

    Returns NaN if no class is score‑able.
    """
    # split off label columns
    sol = solution_df.drop(columns=[row_id])
    sub = submission_df.drop(columns=[row_id])

    aucs = []
    for col in sol.columns:
        if sol[col].sum() == 0:
            continue                    # skip species never present
        try:
            aucs.append(roc_auc_score(sol[col].values, sub[col].values))
        except ValueError:
            # happens if predictions are constant 0/1 – just skip
            pass

    return float(np.mean(aucs)) if aucs else np.nan

# --------------------------------------------------------------------------
# classifier head -----------------------------------------------------------
class Classifier(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, pool: str = "mean"):
        super().__init__()
        self.pool = pool
        self.mlp  = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = x.mean(1) if self.pool == "mean" else x.max(1).values
        return self.mlp(x)

# --------------------------------------------------------------------------
# complete net --------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, encoder_path: str, context_len: int,
                 num_classes: int, pool: str):
        super().__init__()
        self.encoder = load_model(encoder_path, load_weights=True) \
                       if encoder_path else None
        self.classifier = Classifier(64, num_classes, pool)

    def forward(self, spec):                 # spec (B,F,T)
        spec = spec.unsqueeze(1).transpose(2, 3)   # (B,1,T,F)
        emb  = self.encoder.inference_forward(spec)[0]
        return self.classifier(emb)

# --------------------------------------------------------------------------
# trainer -------------------------------------------------------------------
class Trainer:
    def __init__(self, args):
        self.args = args
        self.data = BirdDataModule(args.train_csv,
                                   args.train_spec_dir,
                                   args.val_spec_dir,
                                   args.context_length,
                                   args.batch_size,
                                   num_workers=args.num_workers)

        self.net  = Net(args.pretrained_model_path,
                        args.context_length,
                        self.data.num_classes,
                        args.pool_type)

        self.dev  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.dev)
        self.crit = nn.BCEWithLogitsLoss()
        self.opt  = torch.optim.Adam(
            self.net.classifier.parameters() if args.freeze_encoder
            else self.net.parameters(),
            lr=args.learning_rate)
        self.best_loss = 9e9

        # -------- dirs & metadata -----------------------------------
        self.run_dir     = Path(args.output_dir)
        self.weights_dir = self.run_dir / "weights"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(exist_ok=True)
        (self.run_dir / "config.json").write_text(
            json.dumps(vars(args), indent=2)
        )

        # tee everything
        self.log = Tee(self.run_dir / "train_log.txt")

    # ---------------------------------------------------------------
    def step_batch(self, batch):
        # batch can be (spec, lab, fnames)   or   (spec, tgt, ctx, lab, mask, fnames)
        if len(batch) == 3:                 # new no‑mask collate
            x, _, fnames = batch
        else:                               # legacy masking tuple
            x, _, _, _, _, fnames = batch
        idxs = [self.data.label_idx(f) for f in fnames]
        if -1 in idxs: return None          # skip unlabeled
        y = torch.zeros(len(idxs), self.data.num_classes, device=self.dev)
        for i, c in enumerate(idxs): y[i, c] = 1

        x, y = x.to(self.dev), y.to(self.dev)
        self.opt.zero_grad()
        loss = self.crit(self.net(x), y)
        loss.backward()
        self.opt.step()
        return loss.item()

    # ---------------------------------------------------------------
    def train(self):
        raw_loss, tr_hist, val_hist, steps = [], [], [], []
        impatience = 0
        ema_val = None
        ema_train = None
        alpha = 0.1  # smoothing factor for EMA
        for step in range(1, self.args.max_steps + 1):
            l = self.step_batch(next(iter(self.data.train_loader)))
            if l is None: continue
            raw_loss.append(l)

            if step % self.args.eval_interval: continue
            val = self.evaluate()

            steps.append(step)
            # Compute EMA for train loss
            if ema_train is None:
                ema_train = l
            else:
                ema_train = alpha * l + (1 - alpha) * ema_train
            tr_hist.append(ema_train)
            # Compute EMA for val loss
            if ema_val is None:
                ema_val = val
            else:
                ema_val = alpha * val + (1 - alpha) * ema_val
            val_hist.append(ema_val)
            # ---- grad‑norm -------------------------------------------------
            gn_sq = 0.0
            for p in self.net.parameters():
                if p.grad is not None:
                    gn_sq += p.grad.norm() ** 2
            grad_norm = gn_sq ** 0.5

            self.log(f'step {step:05d} | loss {ema_train:.4f} '
                     f'| val {ema_val:.4f} | grad‑norm {grad_norm:.2f}')

            if val < self.best_loss:                   # improvement
                self.best_loss = val
                impatience = 0
            else:
                impatience += 1
            if step % self.args.save_interval == 0:
                self.save(step, 'ckpt')
            if impatience >= self.args.early_stopping_patience:
                self.log(f'early stop at step {step} '
                         f'(val not improved for {impatience} evals)')
                self.save(step, 'ckpt')
                break

        # ---------- artifacts --------------------------------------
        loss_dict = {"step": steps, "train_loss": tr_hist, "val_loss": val_hist}
        (self.run_dir / "loss.json").write_text(json.dumps(loss_dict, indent=2))

        try:
            plt.figure(figsize=(8, 5))
            plt.plot(steps, tr_hist, label='train')
            plt.plot(steps, val_hist, label='val')
            plt.legend(); plt.xlabel('step'); plt.ylabel('loss'); plt.grid(alpha=.3)
            plt.tight_layout()
            plt.savefig(self.run_dir / "loss_plot.png", dpi=300)
            plt.close()
        except Exception as e:
            print("plotting failed:", e)

    # ---------------------------------------------------------------
    def evaluate(self, n_batches: int = 5):
        self.net.eval(); tot = 0.
        with torch.no_grad():
            for _ in range(n_batches):
                try: batch = next(iter(self.data.val_loader))
                except StopIteration: break
                spec, _, fname = batch
                idx = self.data.label_idx(fname[0])
                if idx == -1: continue
                x = spec.to(self.dev)
                y = torch.zeros(1, self.data.num_classes, device=self.dev)
                y[0, idx] = 1
                tot += self.crit(self.net(x), y).item()
        self.net.train()
        return tot / max(1, n_batches)

    # ---------------------------------------------------------------
    def save(self, step, tag='ckpt'):
        fn = self.weights_dir / f'{tag}_step_{step}.pt'
        torch.save({'step': step,
                    'model_state_dict': self.net.state_dict()}, fn)
        print('saved', fn)

# --------------------------------------------------------------------------
# inference -----------------------------------------------------------------
class Infer:
    def __init__(self, args):
        self.args = args
        # use val directory for both slots; we only need label mapping + files
        self.data = BirdDataModule(args.train_csv,
                                   args.val_spec_dir,
                                   args.val_spec_dir,
                                   context_len=None, batch_size=1,
                                   infinite_train=False, num_workers=0)

        self.classes = self.data.classes

        ckpts = sorted((Path(args.output_dir) / "weights").glob('ckpt*'))
        if not ckpts:
            raise RuntimeError('no ckpt found in weights/')
        ckpt = torch.load(ckpts[-1], map_location='cpu')

        self.net = Net(args.pretrained_model_path, args.context_length,
                       self.data.num_classes, args.pool_type)
        self.net.load_state_dict(ckpt['model_state_dict'])
        self.net.eval()
        self.net.to('cpu')

        onnx_path = Path(args.onnx_model_path or args.output_dir) / 'model.onnx'
        if not onnx_path.exists():
            dummy = torch.randn(1, 513, args.context_length)
            torch.onnx.export(self.net, dummy, onnx_path,
                              input_names=['input'], output_names=['out'],
                              dynamic_axes={'input': {0: 'b'},
                                            'out':   {0: 'b'}},
                              opset_version=13)
        self.session = ort.InferenceSession(str(onnx_path),
                                            providers=['CPUExecutionProvider'])

        # Use log_dir for logs and outputs
        log_dir = getattr(args, 'log_dir', args.output_dir)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log = Tee(Path(log_dir) / "infer_log.txt")

    # ---------------------------------------------------------------
    def _predict_one(self, spec):
        inp = spec.numpy().astype(np.float32)[None]
        out = self.session.run(None, {self.session.get_inputs()[0].name: inp})[0]
        return 1. / (1. + np.exp(-out))  # sigmoid

    # ---------------------------------------------------------------
    def run(self):
        rows, y_true = [], []
        t0 = time.time(); files_done = 0
        log_dir = getattr(self.args, 'log_dir', self.args.output_dir)
        profile_path = Path(log_dir) / "onnx_profile.json"
        self.session.start_profiling(str(profile_path))
        for spec, _, fname in self.data.val_loader:
            # full file → 5‑sec chunks
            segs  = self._segment(spec)                  # (N,F,T)
            probs = [self._predict_one(s) for s in segs] # list of (1,C)

            base  = re.sub(r'_segment_\d+$', '', Path(fname[0]).stem)
            truth = np.zeros(len(self.classes))
            truth[self.data.label_idx(fname[0])] = 1

            for i, pr in enumerate(probs):
                tm   = (i + 1) * 5                       # 5,10,15…
                row_id = f"{base}_{tm}"
                row = {"row_id": row_id}
                row.update({c: pr[0, j] for j, c in enumerate(self.classes)})
                rows.append(row)
                y_true.append(truth)                     # same label for each chunk

            files_done += 1
            if files_done >= 100:
                break
        self.session.end_profiling()

        # ---------- write & score -------------------------------
        sub_df = pd.DataFrame(rows)
        log_dir = getattr(self.args, 'log_dir', self.args.output_dir)
        sub_df.to_csv(Path(log_dir) / self.args.submission_csv, index=False, float_format='%.8f')

        if sub_df.empty:
            # No predictions made, create empty sol_df with correct columns
            sol_df = pd.DataFrame(columns=["row_id"] + list(self.classes))
            print(f"No predictions were made. Wrote empty submission to {Path(log_dir) / self.args.submission_csv}")
            return

        sol_df = pd.DataFrame(y_true, columns=self.classes)
        sol_df.insert(0, "row_id", sub_df["row_id"])

        auc = kaggle_roc_auc(sol_df, sub_df)
        print(f'wrote {Path(log_dir) / self.args.submission_csv}  |  ROC‑AUC = {auc:.4f}')

    # ---------------------------------------------------------------
    @staticmethod
    def _segment(spec, seg=1000):
        T = spec.shape[-1]
        if T % seg:
            spec = F.pad(spec, (0, seg - T % seg))
        return spec.unfold(-1, seg, seg).squeeze(0).permute(1, 0, 2)

# --------------------------------------------------------------------------
# -------- poor‑man's tee ----------------------------------------------------
class Tee:
    """print(msg) will also append to <run_dir>/<fname>"""
    def __init__(self, path):
        self.file = open(path, "a", buffering=1)  # line‑buffered

    def __call__(self, *msg):
        text = " ".join(str(m) for m in msg)
        print(text)
        self.file.write(text + "\n")

    def close(self):
        self.file.close()

# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=['train', 'infer'])
    p.add_argument('--train_spec_dir'); p.add_argument('--val_spec_dir')
    p.add_argument('--train_csv');      p.add_argument('--output_dir', default='.')
    p.add_argument('--pretrained_model_path')
    p.add_argument('--context_length', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--max_steps', type=int, default=10000)
    p.add_argument('--eval_interval', type=int, default=100)
    p.add_argument('--save_interval', type=int, default=250)
    p.add_argument('--early_stopping_patience', type=int, default=100)
    p.add_argument('--pool_type', choices=['mean', 'max'], default='mean')
    p.add_argument('--freeze_encoder', action='store_true')
    p.add_argument('--submission_csv', default='submission.csv')
    p.add_argument('--onnx_model_path')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--log_dir', type=str, help='Directory for inference logs and outputs (defaults to output_dir)')
    args = p.parse_args()

    Path(args.output_dir, 'weights').mkdir(parents=True, exist_ok=True)

    if args.mode == 'train':
        t = Trainer(args); t.train(); t.log.close()
    else:
        # Set log_dir for inference
        if not hasattr(args, 'log_dir') or args.log_dir is None:
            args.log_dir = args.output_dir
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        i = Infer(args); i.run(); i.log.close()

if __name__ == '__main__':
    main()
