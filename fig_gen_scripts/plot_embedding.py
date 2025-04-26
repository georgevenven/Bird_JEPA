#!/usr/bin/env python3
# plot_embedding_by_label.py
#
# CLI unchanged:
#   python plot_embedding_by_label.py --spec_dir … --ckpt_dir …

import argparse, glob, sys, os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ──────────────────────────────────────────────────────────────────────
# repo import
here = Path(__file__).resolve()
repo_root = here.parents[1]                 # Bird_JEPA/
sys.path.insert(0, str(repo_root / "src"))
from models.birdjepa import BirdJEPA, BJConfig            # noqa: E402

try:
    import umap                     # pip install umap-learn
except ImportError:
    sys.exit("UMAP not found – pip install umap-learn and retry.")

# ══════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════
def make_cmap(n_cls=73):
    """discrete 73-colour palette (tab20 + variants)"""
    base = (plt.get_cmap("tab20").colors +
            plt.get_cmap("tab20b").colors +
            plt.get_cmap("tab20c").colors)
    repeats = (n_cls + len(base) - 1) // len(base)
    return ListedColormap((base * repeats)[:n_cls])


def load_encoder(ckpt_dir: Path, cfg: BJConfig, device):
    ckpt = (sorted(ckpt_dir.glob("enc*.pt")) +
            sorted(ckpt_dir.glob("best_*.pt")) +
            list(ckpt_dir.glob("best.pt")))
    if not ckpt:
        raise FileNotFoundError(f"no weights in {ckpt_dir}")

    state = torch.load(ckpt[-1], map_location="cpu")
    state = state.get("enc", state.get("model", state))
    if "model" in state:            # fine-tune format
        state = {k[len("encoder."):] : v
                 for k, v in state.items() if k.startswith("encoder.")}
    enc = BirdJEPA(cfg).to(device)
    enc.load_state_dict(state, strict=False)
    enc.eval()
    return enc

# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_clip(pt_path: Path, enc: BirdJEPA, device):
    """
    Returns (per-token, time-stacked):
        z_tok   : (Tp, Fp*d)   encoder tokens (freq-stacked)
        spec_tok: (Tp, F)      z-scored mel tokens (freq-stacked)
        lab_tok : (Tp,)        integer label per token (time centre)
        mag_ft  : (Fp,Tp)      |z| grid for sanity plot
        spec_z  : (F,T)        z-scored full spectrogram
    """
    d        = torch.load(pt_path, map_location="cpu")
    spec_raw = d["s"].float()                # (F,T_full)
    spec_z   = (spec_raw - spec_raw.mean()) / (spec_raw.std() + 1e-6)
    labels   = d["labels"].long()            # (T_full,)

    z_flat = enc(spec_z[None, None].to(device)).squeeze(0).cpu()  # (Fp*Tp,d)

    # grid dims
    Fp = enc.cfg.n_mels
    for s_f, _ in enc.cfg.conv_str:
        Fp //= s_f
    Tp, d_h = z_flat.shape[0] // Fp, z_flat.shape[1]

    z_ft   = z_flat.view(Fp, Tp, d_h)                      # (Fp,Tp,d)
    mag_ft = z_ft.norm(dim=-1).numpy()                     # (Fp,Tp)

    z_tok  = z_ft.permute(1, 0, 2).reshape(Tp, Fp * d_h)   # (Tp,Fp*d)
    spec_tok = spec_tokens(spec_z, Tp)                     # (Tp,F)
    lab_tok  = labels[np.linspace(0, spec_z.shape[1]-1, Tp, dtype=int)].numpy()

    print(f"{pt_path.name}:  Fp={Fp}, Tp={Tp}, d={d_h}, "
          f"z_tok={z_tok.shape}, spec_tok={spec_tok.shape}")

    return z_tok, spec_tok, lab_tok, mag_ft, spec_z.numpy()

# ──────────────────────────────────────────────────────────────────────
def spec_tokens(spec_z: torch.Tensor, Tp: int) -> np.ndarray:
    """
    Split the full spectrogram into Tp equal-width chunks
    and average over time → (Tp,F)
    """
    F, T = spec_z.shape
    edges = np.linspace(0, T, Tp + 1, dtype=int)
    frames = [spec_z[:, edges[i]:edges[i+1]] for i in range(Tp)]
    frames = [f if f.numel() else spec_z[:, edges[i]:edges[i]+1]   # pad empty
              for i, f in enumerate(frames)]
    return torch.stack([f.mean(dim=1) for f in frames]).numpy()


def sanity_grid(rows, out_png="sanity_check.png"):
    """side-by-side: full spectrogram + |z| grid for up to 10 clips"""
    n = len(rows)
    fig, ax = plt.subplots(n, 2, figsize=(10, 2.2*n), dpi=120,
                           gridspec_kw=dict(wspace=.02, hspace=.25))
    vmin = min(r[2].min() for r in rows)
    vmax = max(r[2].max() for r in rows)

    for i, (name, spec_z, pc1) in enumerate(rows):
        ax[i, 0].imshow(spec_z, aspect="auto", origin="lower", cmap="magma")
        ax[i, 0].set_title(name, fontsize=8); ax[i, 0].axis("off")

        im = ax[i, 1].imshow(pc1, aspect="auto", origin="lower",
                             cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax[i, 1].axis("off")

    cb = fig.colorbar(im, ax=ax[:, 1], fraction=.025, pad=.01)
    cb.set_label("token PC-1")
    plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"sanity-check figure saved to {out_png}")

# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec_dir", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--out_png",  default="umap_embedding.png")
    p.add_argument("--n_mels",   type=int, default=128)
    p.add_argument("--d_model",  type=int, default=192)
    p.add_argument("--max_tokens", type=int, default=25_600)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    cfg    = BJConfig(d_model=args.d_model, n_mels=args.n_mels)
    enc    = load_encoder(Path(args.ckpt_dir), cfg, device)

    pt_files = sorted(glob.glob(os.path.join(args.spec_dir, "*.pt")))
    if not pt_files:
        sys.exit("no .pt files found")

    token_budget  = args.max_tokens
    tokens_so_far = 0

    Z, S, L     = [], [], []
    sanity_rows = []

    for fp in pt_files:
        if tokens_so_far >= token_budget:
            break

        z_tok, s_tok, lab_tok, mag_ft, spec_z = encode_clip(Path(fp), enc, device)
        if lab_tok.max() == 0:
            continue

        keep = min(token_budget - tokens_so_far, z_tok.shape[0])
        Z.append(z_tok[:keep]); S.append(s_tok[:keep]); L.append(lab_tok[:keep])
        tokens_so_far += keep

        if len(sanity_rows) < 10:
            sanity_rows.append((Path(fp).stem, spec_z, mag_ft))

    print(f"loaded {len(Z)} fragments → total frames {tokens_so_far}")

    Z = np.concatenate(Z, 0)
    S = np.concatenate(S, 0)
    L = np.concatenate(L, 0)

    print(f"[raw] S {S.shape}   [enc] Z {Z.shape}   labels {L.shape}")

    reducer_spec = umap.UMAP(n_neighbors=200, min_dist=0.1, metric="euclidean",
                             n_components=2, random_state=42)
    print("\n[UMAP-RAW] fitting …"); emb_spec = reducer_spec.fit_transform(S)

    reducer_enc  = umap.UMAP(n_neighbors=200, min_dist=0.1, metric="euclidean",
                             n_components=2, random_state=42)
    print("\n[UMAP-ENC] fitting …"); emb_enc  = reducer_enc.fit_transform(Z)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6), dpi=150)

    def _scatter(a, emb, lab, title):
        bg = lab == 0; fg = ~bg; lab_fg = lab[fg] - 1
        if bg.any():
            a.scatter(emb[bg,0], emb[bg,1], c="black", s=3, alpha=.35)
        a.scatter(emb[fg,0], emb[fg,1], c=lab_fg, s=3, alpha=.7,
                  cmap=make_cmap(lab.max()))
        a.set_title(title); a.axis("off")

    _scatter(ax[0], emb_spec, L, "raw z-scored log-mel")
    _scatter(ax[1], emb_enc,  L, "BirdJEPA encoder latent")

    plt.tight_layout(); plt.savefig(args.out_png, bbox_inches="tight")
    print(f"wrote {args.out_png}")

    sanity_grid(sanity_rows)

# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
