#!/usr/bin/env python
"""
usage:
  python tools/encode_pca.py \
      --spec_dir  xeno_specs \
      --ckpt      runs/pretrain/weights/best.pt \
      --out       runs/pretrain/enc_pca.png \
      --frags     300   --sample 1.0
"""
import argparse, random, numpy as np, torch, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from matplotlib import cm

from models.birdjepa import BirdJEPA, BJConfig
from data.bird_datasets import load_np

# -----------------------------------------------------------
def grab_fragments(spec_dir, n=300, seg_len=50):
    paths = [p for p in Path(spec_dir).glob('*.npz')]
    random.shuffle(paths)
    frags = []
    for p in paths:
        npz = load_np(p)
        s = npz['s'][:, :-1]          # (F,T)
        if s.shape[1] < seg_len: continue
        t0 = random.randint(0, s.shape[1] - seg_len)
        frags.append(s[:, t0:t0+seg_len])
        if len(frags) >= n: break
    return np.stack(frags)            # (N,F,T)

# -----------------------------------------------------------
def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = BJConfig()
    net = BirdJEPA(cfg).eval().to(dev)
    net.load_state_dict(torch.load(args.ckpt, map_location=dev), strict=False)

    frags = grab_fragments(args.spec_dir, args.frags, cfg.mask_t)    # (N,F,T)
    x = torch.from_numpy(frags).float().unsqueeze(1).transpose(2,3).to(dev)

    with torch.no_grad():
        z = net.encoder(net.stem(x))      # (N, seq, d)
        N, S, D = z.shape
        emb = z.reshape(N*S, D).cpu().numpy()         # every time‑bin -> row

    # optional subsample if too many
    if args.sample < 1.0:
        keep = np.random.rand(len(emb)) < args.sample
        emb = emb[keep]

    # ---- pca ----
    pts = PCA(n_components=2).fit_transform(emb)      # (M,2)

    # ---- plot ----
    plt.figure(figsize=(6,5))
    # colour‑code by parent fragment id to see temporal stripes
    colours = cm.get_cmap('tab20')(np.repeat(np.arange(N), S))[:len(pts)]
    plt.scatter(pts[:,0], pts[:,1], s=4, alpha=.6, c=colours, linewidths=0)
    plt.title('context encoder tokens – pca')
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200)
    plt.close()

# -----------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec_dir', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='enc_pca.png')
    ap.add_argument('--frags', type=int, default=300,
                    help='number of spectrogram fragments to sample')
    ap.add_argument('--sample', type=float, default=1.0,
                    help='keep fraction of tokens (1.0 = all)')
    args = ap.parse_args()
    main(args)
