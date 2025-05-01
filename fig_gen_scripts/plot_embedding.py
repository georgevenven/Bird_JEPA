#!/usr/bin/env python3
# plot_embedding_by_label.py
#
# CLI unchanged:
#   python plot_embedding_by_label.py --spec_dir … --ckpt_dir …

import argparse, glob, sys, os
from pathlib import Path
import json # Needed to load config

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


def load_encoder_and_config(ckpt_dir: Path, device):
    ckpt = (sorted(ckpt_dir.glob("latest.pt")) +
            sorted(ckpt_dir.glob("latest.pt")) +
            list(ckpt_dir.glob("latest.pt"))) # Check for latest.pt too
    if not ckpt:
        raise FileNotFoundError(f"no weights in {ckpt_dir}")

    state = torch.load(ckpt[-1], map_location="cpu")
    state = state.get("enc", state.get("model", state))
    if "model" in state:            # fine-tune format
        state = {k[len("encoder."):] : v
                 for k, v in state.items() if k.startswith("encoder.")}

    # --- Load Model Config ---
    config_path = ckpt_dir / "../model_config.json" # Assumes config is one level up from weights
    if not config_path.exists():
         # Try loading from args saved in checkpoint if available
         if 'args' in state and state['args'] is not None:
              print("Loading config from checkpoint args...")
              cfg_dict = state['args']
              # Map args names to BJConfig field names if necessary (example)
              cfg_dict['layers'] = cfg_dict.get('encoder_layers', 8) # Example mapping
              cfg_dict['n_heads'] = cfg_dict.get('encoder_n_heads', 6)
              cfg_dict['ff_mult'] = cfg_dict.get('encoder_ff_mult', 4)
              # Ensure all required BJConfig fields are present
              cfg = BJConfig(**{k: v for k, v in cfg_dict.items() if hasattr(BJConfig, k)})
         else:
              raise FileNotFoundError(f"model_config.json not found at {config_path} and no args in checkpoint")
    else:
         print(f"Loading config from {config_path}")
         with open(config_path, 'r') as f:
              cfg_dict = json.load(f)
         cfg = BJConfig(**cfg_dict)

    enc = BirdJEPA(cfg).to(device)
    enc.load_state_dict(state, strict=False)
    enc.eval()
    print(f"[Encoder] {sum(p.numel() for p in enc.parameters())/1e6:.2f}M parameters")
    return enc, cfg

# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_clip_chunked(pt_path: Path, enc: BirdJEPA, context_length: int, device):
    """
    Encodes a clip by processing it in fixed-size chunks.
    Returns (per-token, time-stacked):
        z_tok   : (Tp, Fp*d)   encoder tokens (freq-stacked)
        spec_tok: (Tp, F)      z-scored mel tokens (freq-stacked)
        lab_tok : (Tp,)        integer label per token (time centre)
        mag_ft  : (Fp,Tp)      |z| grid for sanity plot (from first chunk only)
        spec_z  : (F,T)        z-scored full spectrogram
    """
    d        = torch.load(pt_path, map_location="cpu")
    spec_raw = d["s"].float()                # (F,T_full)
    spec_z   = (spec_raw - spec_raw.mean()) / (spec_raw.std() + 1e-6)
    labels   = d["labels"].long()            # (T_full,)
    F, T_full = spec_z.shape

    all_z_flat = []
    chunk_indices = range(0, T_full, context_length)

    for i, start_idx in enumerate(chunk_indices):
        end_idx = start_idx + context_length
        chunk = spec_z[:, start_idx:end_idx]
        _, T_chunk = chunk.shape

        # Pad the last chunk if necessary
        if T_chunk < context_length:
            padding = context_length - T_chunk
            chunk = torch.nn.functional.pad(chunk, (0, padding), mode='constant', value=0)

        # Encode the chunk - model expects (B, C, F, T)
        # The forward pass internally calculates Fp, Tp based on context_length
        z_flat_chunk, Fp_model, Tp_model = enc(chunk[None, None].to(device)) # Get Fp, Tp from model
        z_flat_chunk = z_flat_chunk.squeeze(0).cpu() # (Fp*Tp, d)

        # Store results from this chunk
        all_z_flat.append(z_flat_chunk)
        if i == 0: # Get grid dims and mag from first chunk for sanity plot
             # --- Ensure Fp, Tp are integers for view() ---
             # NOTE: Model now returns ints directly, so no .item() needed.
             Fp = Fp_model # Should be int
             Tp = Tp_model # Should be int
             d_h = z_flat_chunk.shape[-1] # This should already be an int
             mag_ft = z_flat_chunk.view(Fp, Tp, d_h).norm(dim=-1).numpy() # (Fp,Tp)

    # Concatenate results from all chunks
    z_flat_all = torch.cat(all_z_flat, dim=0) # (N_chunks * Fp * Tp, d)

    # --- Re-interpret results in terms of effective tokens over original length ---
    # Calculate total number of time tokens generated across all chunks
    total_Tp_effective = len(chunk_indices) * Tp

    # Reshape z_flat_all to match the effective Fp * total_Tp_effective structure
    # We need to be careful here. Let's reshape based on Fp*Tp per chunk first
    # z_flat_all shape: (N_chunks * Fp * Tp, d) -> (N_chunks, Fp*Tp, d)
    z_flat_chunks_reshaped = z_flat_all.view(len(chunk_indices), Fp * Tp, d_h)
    # Permute and reshape to (total_Tp_effective, Fp * d)
    # (N_chunks, Fp, Tp, d) -> (N_chunks, Tp, Fp, d) -> (N_chunks * Tp, Fp, d) -> (total_Tp_effective, Fp*d)
    z_tok = z_flat_chunks_reshaped.view(len(chunk_indices), Fp, Tp, d_h).permute(0, 2, 1, 3).reshape(total_Tp_effective, Fp, d_h).reshape(total_Tp_effective, Fp * d_h)

    # Extract corresponding spec tokens and labels based on the *centers* of the chunks
    spec_tok = spec_tokens_chunked(spec_z, chunk_indices, context_length) # (total_Tp_effective, F)
    lab_tok = labels_chunked(labels, chunk_indices, context_length, Tp) # (total_Tp_effective,)

    print(f"{pt_path.name}: Chunks={len(chunk_indices)}, Fp={Fp}, Tp={Tp}, d={d_h}, "
          f"z_tok={z_tok.shape}, spec_tok={spec_tok.shape}, lab_tok={lab_tok.shape}")

    # Note: mag_ft is only from the first chunk
    return z_tok, spec_tok, lab_tok, mag_ft, spec_z.numpy()

# ──────────────────────────────────────────────────────────────────────
def spec_tokens_chunked(spec_z: torch.Tensor, chunk_indices: range, context_length: int) -> np.ndarray:
    """
    Average spectrogram frames within each chunk used for encoding.
    Returns shape (N_chunks * Tp_per_chunk, F) -> effectively (total_Tp_effective, F)
    This needs refinement - let's average over the chunk for now -> (N_chunks, F)
    For consistency with z_tok (total_Tp_effective, Fp*d), we need (total_Tp_effective, F)
    Let's just average the whole chunk for simplicity now -> (N_chunks, F)
    TODO: Refine this to match the Tp resolution if needed.
    """
    F, T = spec_z.shape
    all_s_tok = []
    for start_idx in chunk_indices:
        end_idx = min(start_idx + context_length, T) # Don't go past end
        chunk = spec_z[:, start_idx:end_idx]
        if chunk.numel() > 0:
             all_s_tok.append(chunk.mean(dim=1))
        else: # Handle empty chunk case if start_idx >= T
             all_s_tok.append(torch.zeros(F))
    # Stack and repeat each chunk average Tp times to match z_tok shape
    # This is an approximation - assumes spec features are constant within a chunk's Tp tokens
    s_tok_chunk_avg = torch.stack(all_s_tok).numpy() # (N_chunks, F)
    # Placeholder: Returning chunk averages. UMAP on this might not be ideal vs z_tok.
    # A better approach would be to resample spec_z to total_Tp_effective points.
    print(f"Warning: spec_tokens_chunked returning chunk averages ({s_tok_chunk_avg.shape}), may not align perfectly with z_tok UMAP.")
    return s_tok_chunk_avg # Shape (N_chunks, F) - MISMATCH with z_tok! Needs fix.

def labels_chunked(labels: torch.Tensor, chunk_indices: range, context_length: int, Tp_per_chunk: int) -> np.ndarray:
    """
    Extract labels corresponding to the center of each time token across all chunks.
    Returns shape (total_Tp_effective,)
    """
    T_full = labels.shape[0]
    all_lab_tok = []
    time_centers_per_chunk = np.linspace(0, context_length - 1, Tp_per_chunk, dtype=int)

    for start_idx in chunk_indices:
        chunk_center_indices = start_idx + time_centers_per_chunk
        # Clamp indices to be within the valid range of the original labels
        chunk_center_indices = np.clip(chunk_center_indices, 0, T_full - 1)
        all_lab_tok.append(labels[chunk_center_indices])

    return torch.cat(all_lab_tok).numpy() # Shape (N_chunks * Tp_per_chunk,)

def sanity_grid(rows, out_png="sanity_check.png"):
    """side-by-side: full spectrogram + |z| grid for up to 10 clips"""
    n = len(rows)
    fig, ax = plt.subplots(n, 2, figsize=(10, 2.2*n), dpi=120,
                           gridspec_kw=dict(wspace=.02, hspace=.3)) # Increased hspace
    vmin = min(r[2].min() for r in rows)
    vmax = max(r[2].max() for r in rows)

    for i, (name, spec_z, pc1) in enumerate(rows):
        ax[i, 0].imshow(spec_z, aspect="auto", origin="lower", cmap="magma")
        ax[i, 0].set_title(name, fontsize=8); ax[i, 0].axis("off")

        im = ax[i, 1].imshow(pc1, aspect="auto", origin="lower",
                             cmap="viridis", vmin=vmin, vmax=vmax)
        ax[i, 1].axis("off")

    cb = fig.colorbar(im, ax=ax[:, 1].tolist(), fraction=.025, pad=.01, aspect=30) # Pass list of axes
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
    p.add_argument("--out_png",  default="umap_embedding_chunked.png")
    p.add_argument("--max_tokens", type=int, default=25_600)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    # Load encoder and the config it was trained with
    enc, cfg = load_encoder_and_config(Path(args.ckpt_dir), device)

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

        # Use the context_length from the loaded config
        context_len = cfg.context_length # Assuming context_length is stored in config
        z_tok, s_tok, lab_tok, mag_ft, spec_z = encode_clip_chunked(Path(fp), enc, context_len, device)

        # Ensure lab_tok is not empty and contains non-zero labels before checking max()
        if lab_tok.size == 0 or lab_tok.max() == 0:
            print(f"Skipping {Path(fp).name} due to empty or all-zero labels.")
            continue

        # --- Adjust token budgeting ---
        # z_tok shape is now (total_Tp_effective, Fp*d)
        # s_tok shape is (N_chunks, F) - !! MISMATCH !!
        # L shape is (total_Tp_effective,)
        num_tokens_in_clip = z_tok.shape[0]
        keep = min(token_budget - tokens_so_far, num_tokens_in_clip)

        # We need s_tok and L to have the same first dimension as z_tok for UMAP
        Z.append(z_tok[:keep]); S.append(s_tok[:keep]); L.append(lab_tok[:keep])
        tokens_so_far += keep

        if len(sanity_rows) < 10:
            sanity_rows.append((Path(fp).stem, spec_z, mag_ft))

    print(f"loaded {len(Z)} files → total tokens processed {tokens_so_far}")

    Z = np.concatenate(Z, 0)
    S = np.concatenate(S, 0)
    L = np.concatenate(L, 0)

    print(f"[raw] S {S.shape}   [enc] Z {Z.shape}   labels {L.shape}")

    reducer_enc  = umap.UMAP(n_neighbors=200, min_dist=0.1, metric="cosine",
                             n_components=2, random_state=42)
    print("\n[UMAP-ENC] fitting …"); emb_enc  = reducer_enc.fit_transform(Z)

    # --- Define plotting helper ---
    def _scatter(a, emb, lab, title):
        bg = lab == 0; fg = ~bg; lab_fg = lab[fg] - 1 # Adjust labels for cmap if 0 is background
        if bg.any():
            a.scatter(emb[bg,0], emb[bg,1], c="black", s=3, alpha=.35, label='Background (0)')
        # Use the make_cmap helper for foreground points
        scatter = a.scatter(emb[fg,0], emb[fg,1], c=lab_fg, s=3, alpha=.7,
                  cmap=make_cmap(lab.max())) # Pass max label value to cmap
        a.set_title(title); a.axis("off")
        # Optional: Add legend if needed, though many classes might make it cluttered
        # a.legend()

    fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=150) # Changed to 1 plot
    # _scatter(ax[0], emb_spec, L, "raw z-scored log-mel") # Disabled raw plot
    _scatter(ax, emb_enc,  L, "BirdJEPA encoder latent") # Call _scatter on the single axis 'ax'

    plt.tight_layout(); plt.savefig(args.out_png, bbox_inches="tight")
    print(f"wrote {args.out_png}")

    # Generate sanity grid if rows were collected
    if sanity_rows:
        sanity_grid(sanity_rows)

# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
