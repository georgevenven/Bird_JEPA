# ──────────────────────────────────────────────────────────────────────────────
# src/pretrain.py      • MAE training with Pixel Reconstruction (Block Mask, No Internal Norm)
# ──────────────────────────────────────────────────────────────────────────────
import argparse, json, time, os, shutil, uuid, datetime as dt, math
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.multiprocessing
import torch.nn.functional as tF        # avoid name clash
import random, copy
import warnings # For eval shape mismatch warning

# Import updated model components
from models.birdjepa import BJConfig, BirdJEPA, SpectrogramDecoder
# Import from utils.py (assuming it's in the same directory)
from utils import load_pretrained_encoder
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
            # Ensure gradient is float32 before norm calculation for stability
            g2+=p.grad.detach().float().norm()**2
    return math.sqrt(g2)

@torch.no_grad()
def param_norm(model: nn.Module) -> float:
    """Computes the overall L2 norm of all parameters in a model."""
    device = next(model.parameters()).device
    total_norm_sq = torch.tensor(0.0, device=device, dtype=torch.float32) # Use float32 for accumulation
    for p in model.parameters():
            total_norm_sq += p.detach().float().norm(2).pow(2) # Use float32 for norm
    return total_norm_sq.sqrt().item()

# --- Utility for converting token mask to pixel mask ---
def token_mask_to_pixel_mask(mask_tok, Fp, Tp, f_stride, t_stride):
    """
    Converts a token mask (B, Fp*Tp) or (Fp*Tp) to a pixel mask.
    If input is (Fp*Tp), it expands to (1, F_bins, T_bins) first.
    Outputs (B, F_bins, T_bins).
    """
    if mask_tok.ndim == 1: # Single mask for the batch
        B = 1
        mask_tok = mask_tok.unsqueeze(0) # Add batch dim
    else:
        B = mask_tok.shape[0]

    F_bins = Fp * f_stride
    T_bins = Tp * t_stride
    device = mask_tok.device

    # Reshape token mask to grid
    mask_tok_grid = mask_tok.view(B, Fp, Tp)

    # Create patch template (used for kronecker product)
    patch_template = torch.ones((f_stride, t_stride), device=device, dtype=torch.bool)

    # Use broadcasting and repeat for efficiency
    mask_tok_expanded = mask_tok_grid.unsqueeze(2).unsqueeze(4) # (B, Fp, 1, Tp, 1)
    patch_template_expanded = patch_template.unsqueeze(0).unsqueeze(1).unsqueeze(3) # (1, 1, f_stride, 1, t_stride)
    pixel_mask_expanded = mask_tok_expanded & patch_template_expanded # (B, Fp, f_stride, Tp, t_stride)
    pixel_mask = pixel_mask_expanded.view(B, Fp, Tp, f_stride, t_stride).permute(0, 1, 3, 2, 4).reshape(B, F_bins, T_bins)

    return pixel_mask

# ---------------------------------------------------------------
AMP = torch.cuda.is_available()
if AMP:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True
    print("AMP Enabled")
else:
    print("AMP Disabled (CUDA not available)")


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
        # Note: Assuming normalization happens in the dataset now.
        print(f"Train dataset size: {len(self.ds)}")
        print(f"Validation dataset size: {len(self.val)}")

        self.train_dl = torch.utils.data.DataLoader(self.ds,  batch_size=args.bs,
                                                    shuffle=True,  num_workers=args.num_workers, # Use arg
                                                    pin_memory=True,  drop_last=True,
                                                    persistent_workers=True if args.num_workers > 0 else False)
        self.val_dl   = torch.utils.data.DataLoader(self.val, batch_size=args.bs,
                                                    shuffle=False, num_workers=args.num_workers, # Use arg
                                                    pin_memory=True, drop_last=False, # Don't drop last for validation
                                                    persistent_workers=True if args.num_workers > 0 else False)

        # ── model  (encoder + decoder) ─────────────────────────
        self.cfg = BJConfig(d_model=args.d_model, n_mels=args.n_mels,
                            layers=args.encoder_layers, # Use arg
                            n_heads=args.encoder_n_heads, # Use arg
                            ff_mult=args.encoder_ff_mult, # Use arg
                            decoder_layers=args.decoder_layers,
                            decoder_d_model=args.decoder_d_model,
                            decoder_n_heads=args.decoder_n_heads,
                            decoder_ff_mult=args.decoder_ff_mult)

        # Use the actual BirdJEPA class which is now our encoder
        self.enc = BirdJEPA(self.cfg)

        # cache token grid dims and strides for fast masking AND decoder init
        self.Fp = self.enc.Fp
        self.Tp = self.enc.Tp
        # Calculate patch size in pixels (frequency and time)
        self.f_stride = self.cfg.n_mels // self.Fp
        self.t_stride = args.context_length // self.Tp
        self.num_pixels_per_patch = self.f_stride * self.t_stride # Assuming 1 channel

        print(f"Spectrogram patch size (pixels): {self.f_stride} (freq) x {self.t_stride} (time)")
        print(f"Token grid size: {self.Fp} (freq) x {self.Tp} (time)")

        # Instantiate the PIXEL SpectrogramDecoder
        self.dec = SpectrogramDecoder(self.cfg,
                                      patch_size_f=self.f_stride,
                                      patch_size_t=self.t_stride,
                                      in_chans=1) # Assuming 1 channel input spec

        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.dev}")

        self.enc.to(self.dev); self.dec.to(self.dev)

        # ── opt / sched ───────────────────────────────────────
        # Combine parameters from encoder and decoder
        all_params = list(self.enc.parameters()) + list(self.dec.parameters())
        self.opt = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.wd) # Use args

        # Cosine decay scheduler with warmup
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.opt, T_max=max(1,args.steps - args.warmup_steps), # Adjust T_max for warmup
                        eta_min=args.lr_min) # Use arg

        self.scaler = torch.cuda.amp.GradScaler(enabled=AMP, init_scale=2.**14) # Adjusted init_scale
        self.crit   = nn.MSELoss(reduction='mean') # Ensure reduction is mean

        # ── run dir / logging ──────────────────────────────────
        self.run_dir = Path(args.run_dir); self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir/'weights').mkdir(exist_ok=True)
        (self.run_dir/'imgs').mkdir(exist_ok=True)
        self.log = Tee(self.run_dir/'train_log.txt')
        # Save config
        if not args.resume: # Only save config on initial run
            with open(self.run_dir / 'config.json', 'w') as f:
                 json.dump(vars(args), f, indent=4)
            with open(self.run_dir / 'model_config.json', 'w') as f:
                 # Use dataclasses.asdict if BJConfig is a dataclass
                 from dataclasses import asdict
                 json.dump(asdict(self.cfg), f, indent=4)


        # track
        self.best_val_loss = float('inf'); self.bad_evals = 0
        self.hist = {'step':[], 'train_loss':[], 'val_loss':[], 'lr':[], 'grad_norm':[]}
        self.train_loss_ema = None
        self.val_loss_ema = None
        self.alpha = 0.1 # EMA smoothing factor

        self._eval_counter = 0
        self.step = 0 # Initialize step counter

        # -------- resume logic ------------------------------------
        if args.resume:
            ckpt_path = self.run_dir/'weights'/'latest.pt' # Standardize checkpoint name
            if ckpt_path.exists():
                print(f'[resume] loading {ckpt_path}')
                pay  = torch.load(ckpt_path, map_location='cpu')
                self.enc.load_state_dict(pay['enc'])
                self.dec.load_state_dict(pay['dec'])
                self.opt.load_state_dict(pay['opt'])
                self.sched.load_state_dict(pay['sched'])
                self.scaler.load_state_dict(pay['scaler'])
                self.step = pay['step']
                self.best_val_loss = pay.get('best_val_loss', float('inf'))
                self.train_loss_ema = pay.get('train_loss_ema', None)
                self.val_loss_ema = pay.get('val_loss_ema', None)
                self.hist = pay.get('hist', self.hist) # Restore history
                print(f'[resume] success: starting from step {self.step+1}')
                self._warm_resumed = True # Flag to handle scheduler stepping
            else:
                print(f'[resume] No checkpoint found at {ckpt_path}, starting from scratch.')
                self._warm_resumed = False
        else:
             self._warm_resumed = False


    # -----------------------------------------------------------------
    # ──────────────────────────────────────────────────────────────
    #  Block Mask Generation - Generates ONE mask for the batch
    # ──────────────────────────────────────────────────────────────
    def _generate_batch_mask(self, F_tokens, T_tokens, mask_ratio=.30, device="cpu"):
        """
        Generates a single block mask for the whole batch using iterative
        rectangle placement until the target mask ratio is reached.
        Operates on the token grid (F_tokens, T_tokens).
        Returns a flat boolean mask (F_tokens * T_tokens,) where True means masked.
        """
        target_masked_tokens = int(mask_ratio * F_tokens * T_tokens)
        masked_count = 0
        # Initialize mask grid (F_tokens, T_tokens)
        mask_grid = torch.zeros(F_tokens, T_tokens, dtype=torch.bool, device=device)

        # Add minimum/maximum block size constraints if desired
        min_h, max_h = self.args.min_mask_h, self.args.max_mask_h
        min_w, max_w = self.args.min_mask_w, self.args.max_mask_w

        attempts = 0 # Safety break
        max_attempts = F_tokens * T_tokens * 2 # Heuristic limit

        while masked_count < target_masked_tokens and attempts < max_attempts:
            attempts += 1
            # Determine block size (height h, width w)
            # Ensure max size doesn't exceed grid dimensions
            h = random.randint(min_h, min(F_tokens, max_h))
            w = random.randint(min_w, min(T_tokens, max_w))

            # Determine top-left corner (f0, t0)
            f0 = random.randint(0, F_tokens - h)
            t0 = random.randint(0, T_tokens - w)

            # Check how many *new* tokens this block would mask
            current_block_mask = torch.zeros_like(mask_grid)
            current_block_mask[f0 : f0 + h, t0 : t0 + w] = True
            newly_masked = torch.logical_and(current_block_mask, ~mask_grid) # Find tokens that are True in block but False in main mask
            num_newly_masked = newly_masked.sum().item()

            # Calculate how many more tokens we *need*
            needed = target_masked_tokens - masked_count

            # Only apply the block if it adds new tokens and doesn't overshoot the target too much
            # (or if we are far from the target and need to add something)
            # Allow slight overshoot by accepting if num_newly_masked is not drastically larger than needed
            if num_newly_masked > 0 and (num_newly_masked <= needed * 1.5 or masked_count < target_masked_tokens * 0.8):
                 # Apply the mask: set the block region to True
                 mask_grid[f0 : f0 + h, t0 : t0 + w] = True
                 # Recalculate the total masked count accurately
                 masked_count = mask_grid.sum().item()

        # If overshoot happened, randomly unmask some tokens to reach the target
        if masked_count > target_masked_tokens:
             excess = masked_count - target_masked_tokens
             # Get indices of currently masked tokens
             masked_indices = torch.nonzero(mask_grid.flatten()).squeeze()
             # Randomly select 'excess' indices to unmask
             unmask_indices = masked_indices[torch.randperm(masked_indices.numel(), device=device)[:excess]]
             # Flatten mask, unmask selected indices, reshape back
             mask_flat = mask_grid.flatten()
             mask_flat[unmask_indices] = False
             # mask_grid = mask_flat.view(F_tokens, T_tokens) # Reshape if needed later
             return mask_flat # Return the final flat mask

        # Return the flat mask (True where masked)
        return mask_grid.flatten()


    # -----------------------------------------------------------------
    def _step(self, spec_batch, is_train=True):
        """ Performs a single training or evaluation step """
        # spec_batch: (B, F, T) from DataLoader - Assume already normalized by dataloader
        spec_batch = spec_batch.to(self.dev)
        B, F_bins, T_bins = spec_batch.shape

        # Add channel dimension: (B, 1, F, T) - Use original spec_batch
        spec_input = spec_batch.unsqueeze(1)

        # --- Generate ONE Token Mask (Block Strategy) for the Batch ---
        Fp, Tp = self.Fp, self.Tp
        num_tokens = Fp * Tp
        # Generate a single block mask (Fp*Tp,)
        single_mask_tok_flat = self._generate_batch_mask(Fp, Tp, self.args.mask_ratio, device=self.dev)
        # Repeat the same mask for all items in the batch
        mask_tok = single_mask_tok_flat.unsqueeze(0).expand(B, -1) # (B, Fp*Tp)

        # --- Calculate indices based on the single mask ---
        # These are now the same for every item in the batch conceptually
        masked_indices = torch.nonzero(single_mask_tok_flat).squeeze() # Indices of masked tokens (1D)
        visible_indices = torch.nonzero(~single_mask_tok_flat).squeeze() # Indices of visible tokens (1D)
        num_masked = masked_indices.numel()
        num_visible = visible_indices.numel()

        # Ensure indices are tensors even if only one element
        if num_masked == 1: masked_indices = masked_indices.unsqueeze(0)
        if num_visible == 1: visible_indices = visible_indices.unsqueeze(0)

        # --- MAE Encoder Forward Pass (Only Visible Tokens) ---
        # This requires modifying the encoder or simulating it.
        # Simulation: Encode all, then select visible based on indices.
        with torch.cuda.amp.autocast(enabled=AMP):
            encoded_all, _, _ = self.enc(spec_input) # (B, Fp*Tp, D_enc)
            D_enc = encoded_all.shape[-1]

            # Select only VISIBLE encoded tokens using indices
            # Expand visible_indices for batch gathering
            visible_indices_batch = visible_indices.unsqueeze(0).expand(B, -1) # (B, num_visible)
            encoded_vis = torch.gather(encoded_all, dim=1, index=visible_indices_batch.unsqueeze(-1).expand(-1, -1, D_enc))
            # encoded_vis shape: (B, num_visible, D_enc)

            # -------- Decoder predicts masked pixels --------------------
            # Pass the boolean mask `mask_tok` (True where masked)
            pred_pixels_flat = self.dec(encoded_vis, mask_tok, (Fp, Tp))
            # pred_pixels_flat shape: (B * num_masked, num_pixels_per_patch)

            # -------- Get target PIXELS for masked tokens -------------
            # Extract patches from the original (dataloader-normalized) spec_input
            spec_patches = spec_input.unfold(2, self.f_stride, self.f_stride).unfold(3, self.t_stride, self.t_stride)
            spec_patches = spec_patches.permute(0, 2, 3, 1, 4, 5).reshape(B, Fp * Tp, -1)
            # Select the target patches using the boolean mask `mask_tok`
            target_pixels_flat = spec_patches[mask_tok]
            # target_pixels_flat shape: (B * num_masked, num_pixels_per_patch)

            # -------- Loss calculation ------------------------------------
            # Ensure float32 for loss calculation stability
            loss = self.crit(pred_pixels_flat.float(), target_pixels_flat.float())

        # --- Backpropagation ---
        gnorm_total = 0.0 # Initialize gradient norm
        if is_train:
            self.opt.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            grad_norm_enc = torch.nn.utils.clip_grad_norm_(self.enc.parameters(), self.args.clip_grad)
            grad_norm_dec = torch.nn.utils.clip_grad_norm_(self.dec.parameters(), self.args.clip_grad)
            gnorm_total = float(grad_norm_enc + grad_norm_dec) # Or use torch.norm if needed
            self.scaler.step(self.opt)
            self.scaler.update()

        # --- Visualization Data Prep ---
        viz_data = None
        # Prepare viz data less frequently if needed, e.g., self.step % (self.args.eval_interval * 5) == 0
        if not is_train or (self.step % self.args.eval_interval == 0):
            if B > 0 and num_masked > 0: # Ensure there's something to visualize
                 with torch.no_grad():
                    # Use the first sample and the common mask for visualization
                    spec_orig_0 = spec_batch[0].cpu() # Original spec for sample 0
                    mask_tok_0 = mask_tok[0].cpu()    # Token mask for sample 0 (same as others)

                    # Reshape predictions and targets for the first sample
                    pred_pixels_sample0 = pred_pixels_flat[:num_masked].detach().cpu()
                    target_pixels_sample0 = target_pixels_flat[:num_masked].detach().cpu()

                    viz_data = {
                        "spec_orig": spec_orig_0,
                        "spec_norm": spec_orig_0, # Pass original again for 'normalized' slot
                        "mask_tok": mask_tok_0,
                        "pred_pixels_flat": pred_pixels_sample0,
                        "target_pixels_flat": target_pixels_sample0,
                        "grid_shape": (Fp, Tp),
                        "patch_shape": (self.f_stride, self.t_stride)
                    }

        return float(loss.item()), gnorm_total, viz_data

    # -----------------------------------------------------------------
    @torch.no_grad()
    def _eval(self):
        """ Evaluate the model on the validation set (single batch) """
        self.enc.eval()
        self.dec.eval()
        avg_loss = float('inf') # Default value
        start_time = time.time()

        try:
            spec_batch, *_ = next(iter(self.val_dl))   # Get ONE batch
            spec_batch = spec_batch.to(self.dev)
            B, F_bins, T_bins = spec_batch.shape
            spec_input = spec_batch.unsqueeze(1)

            if B == 0:
                 print("--- Eval Warning: Validation batch is empty. ---")
                 return float('inf')

            # --- Generate ONE Mask for the Batch ---
            Fp, Tp = self.Fp, self.Tp
            num_tokens = Fp * Tp
            single_mask_tok_flat = self._generate_batch_mask(Fp, Tp, self.args.mask_ratio, device=self.dev)
            mask_tok = single_mask_tok_flat.unsqueeze(0).expand(B, -1)
            masked_indices = torch.nonzero(single_mask_tok_flat).squeeze()
            visible_indices = torch.nonzero(~single_mask_tok_flat).squeeze()
            num_masked = masked_indices.numel()
            num_visible = visible_indices.numel()
            if num_masked == 1: masked_indices = masked_indices.unsqueeze(0)
            if num_visible == 1: visible_indices = visible_indices.unsqueeze(0)

            # --- Forward Pass ---
            with torch.cuda.amp.autocast(enabled=AMP):
                encoded_all, _, _ = self.enc(spec_input.to(torch.half)) # Cast input
                D_enc = encoded_all.shape[-1]

                visible_indices_batch = visible_indices.unsqueeze(0).expand(B, -1)
                encoded_vis = torch.gather(encoded_all, dim=1, index=visible_indices_batch.unsqueeze(-1).expand(-1, -1, D_enc))

                pred_pixels_flat = self.dec(encoded_vis, mask_tok, (Fp, Tp))

                spec_patches = spec_input.unfold(2, self.f_stride, self.f_stride).unfold(3, self.t_stride, self.t_stride)
                spec_patches = spec_patches.permute(0, 2, 3, 1, 4, 5).reshape(B, Fp * Tp, -1)
                target_pixels_flat = spec_patches[mask_tok]

                # --- Loss ---
                if num_masked > 0: # Avoid loss calculation if nothing was masked
                    loss = self.crit(pred_pixels_flat.float(), target_pixels_flat.float())
                    avg_loss = loss.item()
                else:
                    avg_loss = 0.0 # No loss if no tokens were masked
                    print("--- Eval Warning: No tokens were masked in validation batch. ---")


        except StopIteration:
            print("--- Eval Warning: Validation DataLoader is empty. ---")
            avg_loss = float('inf')
        except Exception as e:
             print(f"--- Eval Error: An exception occurred during validation: {e} ---")
             import traceback
             traceback.print_exc()
             avg_loss = float('inf')

        eval_time = time.time() - start_time
        print(f"Validation (1 batch) finished in {eval_time:.2f}s, Loss: {avg_loss:.4f}")

        self.enc.train() # Set back to train mode
        self.dec.train()
        return avg_loss

    # -----------------------------------------------------------------
    # --- train() method remains largely the same ---
    # --- (ensure LR schedule, logging, checkpointing, plotting are correct) ---
    def train(self):
        """ Main training loop """
        start_time = time.time()
        print(f"Starting training from step {self.step}")
        dl_iter = iter(self.train_dl) # Initialize iterator outside loop

        while self.step < self.args.steps:
            # --- Warmup Phase & LR Scheduling ---
            if self.step < self.args.warmup_steps:
                 lr_scale = min(1.0, float(self.step + 1) / self.args.warmup_steps)
                 current_lr_target = self.args.lr * lr_scale
            else:
                 # Cosine decay phase
                 progress = float(self.step - self.args.warmup_steps) / float(max(1, self.args.steps - self.args.warmup_steps))
                 current_lr_target = self.args.lr_min + 0.5 * (self.args.lr - self.args.lr_min) * (1 + math.cos(math.pi * progress))

            # Apply LR to optimizer param groups
            for pg in self.opt.param_groups:
                 pg['lr'] = current_lr_target

            # --- Get Batch ---
            try:
                spec_batch, *_ = next(dl_iter)
            except StopIteration:
                print("Epoch finished. Resetting DataLoader iterator...")
                dl_iter = iter(self.train_dl) # Reset iterator
                spec_batch, *_ = next(dl_iter)

            # --- Training Step ---
            self.step += 1
            train_loss, gnorm_total, viz_data = self._step(spec_batch, is_train=True)

            # --- Logging & Evaluation ---
            if self.step % self.args.eval_interval == 0 or self.step == self.args.steps:
                val_loss = self._eval() # Evaluate on a single batch

                # EMA Loss Calculation
                if self.train_loss_ema is None:
                    self.train_loss_ema = train_loss
                    self.val_loss_ema = val_loss if not math.isinf(val_loss) else train_loss # Handle potential initial Inf val_loss
                else:
                    self.train_loss_ema = self.alpha * train_loss + (1 - self.alpha) * self.train_loss_ema
                    if not math.isinf(val_loss):
                         self.val_loss_ema = self.alpha * val_loss + (1 - self.alpha) * self.val_loss_ema

                current_lr = self.opt.param_groups[0]['lr']
                self.hist['step'].append(self.step)
                self.hist['train_loss'].append(train_loss)
                self.hist['val_loss'].append(val_loss)
                self.hist['lr'].append(current_lr)
                self.hist['grad_norm'].append(gnorm_total)

                elapsed_time = time.time() - start_time
                steps_per_sec = self.args.eval_interval / elapsed_time if elapsed_time > 0 else 0
                eta_seconds = (self.args.steps - self.step) / steps_per_sec if steps_per_sec > 0 else 0
                eta_str = str(dt.timedelta(seconds=int(eta_seconds)))

                msg = (f"step {self.step:07d}/{self.args.steps} | "
                       f"train_loss {train_loss:.4f} (ema {self.train_loss_ema:.4f}) | "
                       f"val_loss {val_loss:.4f} (ema {self.val_loss_ema:.4f}) | "
                       f"lr {current_lr:.2e} | gnorm {gnorm_total:.2e} | "
                       f"{steps_per_sec:.2f} steps/s | eta {eta_str}")
                self.log(msg)

                # --- Visualization ---
                if viz_data:
                    _viz_triptych(
                        self.run_dir/'imgs', f"step_{self.step:07d}",
                        viz_data=viz_data
                    )

                # --- Checkpointing ---
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.bad_evals = 0
                    print(f"  New best val_loss: {self.best_val_loss:.4f}")
                else:
                    self.bad_evals += 1

                # Save latest checkpoint
                ckpt_path = self.run_dir/'weights'/'latest.pt'
                payload = {
                    'step': self.step,
                    'enc': self.enc.state_dict(),
                    'dec': self.dec.state_dict(),
                    'opt': self.opt.state_dict(),
                    'sched': self.sched.state_dict(),
                    'scaler': self.scaler.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'train_loss_ema': self.train_loss_ema,
                    'val_loss_ema': self.val_loss_ema,
                    'hist': self.hist,
                    'args': vars(self.args)
                }
                torch.save(payload, ckpt_path)

                if is_best:
                    best_path = self.run_dir/'weights'/'best.pt'
                    shutil.copyfile(ckpt_path, best_path)
                    print(f"  Saved best checkpoint to {best_path}")

                # Early stopping check
                if self.args.early_stop > 0 and self.bad_evals >= self.args.early_stop:
                    self.log(f"Early stopping triggered after {self.args.early_stop} evaluations without improvement.")
                    break

                start_time = time.time() # Reset timer for next interval

        # --- Final Actions ---
        self.log("Training finished.")
        # Final plot (remains the same)
        df = pd.DataFrame(self.hist)
        df.to_csv(self.run_dir/'metrics.csv', index=False)
        try:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(df.step, df.train_loss, label='Train Loss', alpha=0.7)
            plt.plot(df.step, df.val_loss, label='Val Loss', alpha=0.7)
            # Add EMA plots if available
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss Curve')
            plt.grid(True, alpha=0.5)
            plt.ylim(bottom=0)
            plt.subplot(1, 2, 2)
            plt.plot(df.step, df.lr, label='Learning Rate')
            plt.xlabel('Step')
            plt.ylabel('Learning Rate')
            plt.legend()
            plt.title('Learning Rate Schedule')
            plt.grid(True, alpha=0.5)
            plt.tight_layout()
            plt.savefig(self.run_dir/'curves.png', dpi=150)
            plt.close()
            self.log(f"Saved final metric plots to {self.run_dir/'curves.png'}")
        except Exception as e:
             self.log(f"Failed to generate final plots: {e}")
        self.log.close()


# ───────────────────────────── viz util (Remains mostly the same) ──────────────────────────────
def _viz_triptych(run_dir, step_str, viz_data):
    """
    Visualizes input, masked input, and reconstruction for pixel MAE.
    Assumes input spec is already potentially normalized by dataloader.
    """
    import matplotlib.pyplot as plt
    import torch.nn.functional as tF
    import numpy as np
    import torch

    try:
        # --- Extract data ---
        spec_orig = viz_data["spec_orig"] # This is the potentially dataloader-normalized spec
        mask_tok = viz_data["mask_tok"]   # Boolean token mask (Fp*Tp,)
        pred_pixels_flat = viz_data["pred_pixels_flat"] # (num_masked, P*P)
        Fp, Tp = viz_data["grid_shape"]
        f_stride, t_stride = viz_data["patch_shape"]
        F_bins, T_bins = spec_orig.shape
        num_pixels_per_patch = f_stride * t_stride
        num_masked = pred_pixels_flat.shape[0]

        # --- Create Pixel Mask for Visualization ---
        pixel_mask = token_mask_to_pixel_mask(mask_tok.unsqueeze(0), Fp, Tp, f_stride, t_stride).squeeze(0)
        pixel_mask = pixel_mask.cpu()

        # --- Create Masked Input Visualization ---
        spec_masked_viz = spec_orig.clone()
        mask_fill_value = spec_orig.min() - (spec_orig.max() - spec_orig.min()) * 0.1
        spec_masked_viz[pixel_mask] = mask_fill_value

        # --- Reconstruct Spectrogram Image ---
        recon_spec = spec_orig.clone() # Start with original visible patches

        if num_masked > 0:
            pred_patches = pred_pixels_flat.reshape(num_masked, 1, f_stride, t_stride)
            masked_indices_flat = torch.nonzero(mask_tok).squeeze(-1)
            rows, cols = np.unravel_index(masked_indices_flat.numpy(), (Fp, Tp))
            for i, (r, c) in enumerate(zip(rows, cols)):
                f_start, f_end = r * f_stride, (r + 1) * f_stride
                t_start, t_end = c * t_stride, (c + 1) * t_stride
                recon_spec[f_start:f_end, t_start:t_end] = pred_patches[i, 0]

        # --- Calculate Pixel MSE Map (Optional) ---
        mse_map = torch.zeros_like(spec_orig)
        if num_masked > 0:
            target_pixels_flat = viz_data["target_pixels_flat"]
            target_patches = target_pixels_flat.reshape(num_masked, 1, f_stride, t_stride)
            mse_per_patch = F.mse_loss(pred_patches, target_patches, reduction='none').mean(dim=(1, 2, 3))
            for i, (r, c) in enumerate(zip(rows, cols)):
                 f_start, f_end = r * f_stride, (r + 1) * f_stride
                 t_start, t_end = c * t_stride, (c + 1) * t_stride
                 mse_map[f_start:f_end, t_start:t_end] = mse_per_patch[i]
        mse_map[~pixel_mask] = 0.0

        # ---------- Plotting ----------------------------------------------------
        fig, ax = plt.subplots(2, 3, figsize=(15, 8), dpi=120)
        kw_spec = dict(aspect='auto', origin='lower', cmap='magma')
        kw_err = dict(aspect='auto', origin='lower', cmap='coolwarm', interpolation='nearest')
        kw_diff = dict(aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')

        spec_orig_float = spec_orig.float()
        vmin = torch.quantile(spec_orig_float, 0.01)
        vmax = torch.quantile(spec_orig_float, 0.99)
        if vmin == vmax:
            vmin = spec_orig_float.min()
            vmax = spec_orig_float.max()
            if vmin == vmax: vmax = vmin + 1e-6

        # Row 1
        ax[0,0].imshow(spec_orig, **kw_spec, vmin=vmin, vmax=vmax)
        ax[0,0].set_title('Input Spectrogram')
        ax[0,1].imshow(spec_masked_viz, **kw_spec, vmin=vmin, vmax=vmax)
        ax[0,1].set_title('Masked Input')
        im_err = ax[0,2].imshow(mse_map, **kw_err)
        ax[0,2].set_title('Pixel MSE in Masked Areas')
        fig.colorbar(im_err, ax=ax[0,2], fraction=0.046, pad=0.04)

        # Row 2
        ax[1,0].imshow(spec_orig, **kw_spec, vmin=vmin, vmax=vmax)
        ax[1,0].set_title('Target Spectrogram')
        im_recon = ax[1,1].imshow(recon_spec, **kw_spec, vmin=vmin, vmax=vmax)
        ax[1,1].set_title('Reconstructed Spectrogram')
        fig.colorbar(im_recon, ax=ax[1,1], fraction=0.046, pad=0.04)

        diff_map = (spec_orig - recon_spec).abs()
        im_diff = ax[1,2].imshow(diff_map, **kw_diff)
        ax[1,2].set_title('Absolute Difference')
        fig.colorbar(im_diff, ax=ax[1,2], fraction=0.046, pad=0.04)

        for row_ax in ax:
            for subplot_ax in row_ax:
                 subplot_ax.axis('off')

        fig.suptitle(f"Visualization at {step_str}", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])

        save_path = Path(run_dir) / f"viz_{step_str}.png"
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    except Exception as e:
        print(f"Error during visualization at step {step_str}: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals() and fig is not None:
             plt.close(fig)


# ╭───────────────────────────────────────────────────────────────╮
# │ CLI Arguments                                                │
# ╰───────────────────────────────────────────────────────────────╯
def main():
    p = argparse.ArgumentParser(description="MAE Pretraining with Pixel Reconstruction")
    # Paths
    p.add_argument('--train_dir', type=str, required=True, help="Directory for training spectrograms")
    p.add_argument('--val_dir', type=str, default=None, help="Directory for validation spectrograms (optional, uses train_dir if None)")
    p.add_argument('--run_dir',  type=str, default='runs/mae_pixel_pretrain', help="Directory to save runs, logs, weights")

    # Training HParams
    p.add_argument('--steps',    type=int, default=300000, help="Total training steps")
    p.add_argument('--warmup_steps', type=int, default=10000, help="Number of linear warmup steps")
    p.add_argument('--bs',       type=int, default=64, help="Batch size per GPU")
    p.add_argument('--lr',       type=float, default=5e-4, help="Base learning rate (before warmup/decay)")
    p.add_argument('--lr_min',   type=float, default=1e-5, help="Minimum learning rate for cosine decay")
    p.add_argument('--wd',       type=float, default=0.05, help="Weight decay")
    p.add_argument('--mask_ratio', type=float, default=0.75, help="Fraction of patches to mask")
    p.add_argument('--clip_grad', type=float, default=1.0, help="Gradient clipping value")
    # Block Masking Specific Params
    p.add_argument('--min_mask_h', type=int, default=1, help="Min height of masking block (tokens)")
    p.add_argument('--max_mask_h', type=int, default=4, help="Max height of masking block (tokens)")
    p.add_argument('--min_mask_w', type=int, default=10, help="Min width of masking block (tokens)")
    p.add_argument('--max_mask_w', type=int, default=80, help="Max width of masking block (tokens)")


    # Model Config
    p.add_argument('--context_length', type=int, default=256, help="Spectrogram time length (frames)")
    p.add_argument('--n_mels', type=int, default=128, help='Input spectrogram height (mel bins)')
    # Encoder specific
    p.add_argument('--encoder_layers', type=int, default=8, help="Number of encoder transformer blocks")
    p.add_argument('--d_model',  type=int, default=192, help="Encoder hidden dimension")
    p.add_argument('--encoder_n_heads', type=int, default=6, help="Number of encoder attention heads")
    p.add_argument('--encoder_ff_mult', type=int, default=4, help="Encoder feedforward multiplier")
    # Decoder specific
    p.add_argument('--decoder_layers', type=int, default=4, help="Number of decoder transformer blocks")
    p.add_argument('--decoder_d_model', type=int, default=None, help="Decoder hidden dim (defaults to encoder d_model / 2)")
    p.add_argument('--decoder_n_heads', type=int, default=4, help="Number of decoder attention heads")
    p.add_argument('--decoder_ff_mult', type=int, default=4, help="Decoder feedforward multiplier")

    # System / Logging
    p.add_argument('--num_workers', type=int, default=4, help="Number of DataLoader workers")
    p.add_argument('--eval_interval', type=int, default=500, help="Steps between validation/logging/viz")
    p.add_argument('--early_stop', type=int, default=0, help="Stop after N evals without improvement (0 to disable)")
    p.add_argument('--resume', action='store_true', help='Continue training from latest.pt in run_dir')

    args = p.parse_args()

    # Default decoder dim if not specified
    if args.decoder_d_model is None:
        args.decoder_d_model = args.d_model // 2 # Common practice for MAE
        print(f"Decoder dimension not specified, defaulting to {args.decoder_d_model}")

    # Handle run directory rotation
    rd = Path(args.run_dir)
    if rd.exists() and not args.resume:
        print(f"Run directory {rd} exists and not resuming. Archiving...")
        stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
        archive_dir = Path('runs/archive')
        archive_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(rd), str(archive_dir / f'{rd.name}_{stamp}'))
            print(f"Archived previous run to {archive_dir / f'{rd.name}_{stamp}'}")
            rd.mkdir(parents=True, exist_ok=True) # Recreate after moving
        except Exception as e:
            print(f"Error archiving run directory: {e}")
            exit(1)
    elif not rd.exists():
        rd.mkdir(parents=True, exist_ok=True)

    # Start training
    Trainer(args).train()

if __name__ == '__main__':
    main()
