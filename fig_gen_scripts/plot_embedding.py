#!/usr/bin/env python3
# plot_embedding_by_label.py
#
# every .pt file is a dict:
#   {"s": (F,T) float32  mel-spec,
#    "labels": (T,) int  per-time-bin label }
#
# we flatten the encoder tokens (one per patch) and inherit the majority
# label inside each patch. tokens are then pushed through PCA→2-D and
# rendered with a discrete colormap (73 labels → tab20 + tab20b + tab20c).

import argparse, glob, os, sys
from pathlib import Path

import numpy as np
import torch, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

# add <repo_root>/src  (works no matter where you call the script)
here = Path(__file__).resolve()
repo_root = here.parents[1]          # Bird_JEPA/
sys.path.insert(0, str(repo_root / "src"))
from models.birdjepa import BirdJEPA, BJConfig                                   # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
def load_encoder(ckpt_dir: Path, cfg: BJConfig, device: torch.device):
    ckpt = sorted(ckpt_dir.glob("enc*.pt")) \
         + sorted(ckpt_dir.glob("best_*.pt")) \
         + list(ckpt_dir.glob("best.pt"))
    if not ckpt:
        raise FileNotFoundError(f"no weights in {ckpt_dir}")
    state = torch.load(ckpt[-1], map_location="cpu")
    state = state.get("enc", state.get("model", state))
    if "model" in state:             # fine-tune format
        state = {k[len("encoder."):] : v
                 for k,v in state.items() if k.startswith("encoder.")}
    enc = BirdJEPA(cfg).to(device)
    enc.load_state_dict(state, strict=False); enc.eval()
    return enc

# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_pt(path: Path, enc: BirdJEPA, device: torch.device):
    d = torch.load(path, map_location="cpu")
    spec   = d["s"].float()                   # (F,T_full)
    labels = d["labels"].long()               # (T_full,)
    B,F,T_full = 1,*spec.shape

    # ---- run encoder --------------------------------------------------
    z = enc(spec.unsqueeze(0).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()  # (T_tok,d)

    # ---- map each token to the nearest original frame -----------------
    T_tok = z.shape[0]
    centres = np.linspace(0, T_full-1, T_tok, dtype=int)   # monotone, len==T_tok
    lab_tok = labels[centres].numpy()                     # (T_tok,)

    return z, lab_tok

# ──────────────────────────────────────────────────────────────────────────────
def make_cmap(n_cls=73):
    base = plt.get_cmap('tab20').colors \
         + plt.get_cmap('tab20b').colors \
         + plt.get_cmap('tab20c').colors
    repeats = (n_cls + len(base) - 1)//len(base)
    return ListedColormap((base*repeats)[:n_cls])

# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_png",  default="pca_tokens.png")
    ap.add_argument("--n_mels",   type=int, default=128)
    ap.add_argument("--d_model",  type=int, default=192)
    ap.add_argument("--max_tokens", type=int, default=200_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = BJConfig(d_model=args.d_model, n_mels=args.n_mels)
    device = torch.device(args.device)
    enc = load_encoder(Path(args.ckpt_dir), cfg, device)

    files = sorted(glob.glob(os.path.join(args.spec_dir,"*.pt")))
    if not files: sys.exit("no .pt files")

    all_z, all_lab = [], []
    for fp in files:
        z, lab = encode_pt(Path(fp), enc, device)
        all_z.append(z); all_lab.append(lab)
    z = np.concatenate(all_z,0); lab = np.concatenate(all_lab,0)
    if z.shape[0] > args.max_tokens:
        idx = np.random.choice(z.shape[0], args.max_tokens, replace=False)
        z, lab = z[idx], lab[idx]

    xy = PCA(n_components=2, random_state=0).fit_transform(z)

    plt.figure(figsize=(7,7),dpi=150)
    plt.scatter(xy[:,0], xy[:,1], c=lab, s=3, cmap=make_cmap(lab.max()+1), alpha=.7)
    plt.axis('off'); plt.tight_layout(); plt.savefig(args.out_png, bbox_inches="tight")
    print("wrote", args.out_png)

if __name__=="__main__":
    main()
