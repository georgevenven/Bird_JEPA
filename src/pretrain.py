# ──────────────────────────────────────────────────────────────────────────────
# src/pretrain.py   
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
    if mask_tok.ndim == 1:
        B = 1
        mask_tok = mask_tok.unsqueeze(0)
    else:
        B = mask_tok.shape[0]

    F_bins = Fp * f_stride
    T_bins = Tp * t_stride
    device = mask_tok.device

    mask_tok_grid = mask_tok.view(B, Fp, Tp)

    # Repeat vertically (frequency)
    pixel_mask_freq_repeated = mask_tok_grid.repeat_interleave(f_stride, dim=1)
    # Shape: (B, Fp * f_stride, Tp) == (B, F_bins, Tp)

    # Repeat horizontally (time)
    pixel_mask = pixel_mask_freq_repeated.repeat_interleave(t_stride, dim=2)
    # Shape: (B, F_bins, Tp * t_stride) == (B, F_bins, T_bins)

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
    #  rectangle mask generation (remains the same) - Generates ONE mask
    # ──────────────────────────────────────────────────────────────
    def _generate_batch_mask(self, F_tokens, T_tokens, mask_ratio=.30, device="cpu"):
        """ Generates a single random mask for the whole batch """
        num_tokens = F_tokens * T_tokens
        num_masked = int(mask_ratio * num_tokens)

        # Generate random indices to mask ONCE
        indices = torch.randperm(num_tokens, device=device)
        masked_indices = indices[:num_masked]

        # Create boolean mask (True where masked)
        mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        mask[masked_indices] = True
        # Keep it flat (Fp*Tp,)

        return mask

    # -----------------------------------------------------------------
    def _step(self, spec_batch, is_train=True):
        """ Performs a single training or evaluation step """
        # spec_batch: (B, F, T) from DataLoader - Assume already normalized by dataloader
        spec_batch = spec_batch.to(self.dev)
        B, F_bins, T_bins = spec_batch.shape

        # Add channel dimension: (B, 1, F, T) - Use original spec_batch
        spec_input = spec_batch.unsqueeze(1)

        # --- Generate ONE Token Mask for the Batch ---
        Fp, Tp = self.Fp, self.Tp
        num_tokens = Fp * Tp
        # Generate a single mask (Fp*Tp,)
        single_mask_tok_flat = self._generate_batch_mask(Fp, Tp, self.args.mask_ratio, device=self.dev)
        # Repeat the same mask for all items in the batch
        mask_tok = single_mask_tok_flat.unsqueeze(0).expand(B, -1) # (B, Fp*Tp)

        # --- Calculate indices based on the single mask ---
        # These are now the same for every item in the batch conceptually
        num_masked = single_mask_tok_flat.sum().item()
        num_visible = num_tokens - num_masked
        ids_shuffle = torch.argsort(torch.rand(num_tokens, device=self.dev)) # Need a consistent shuffle order
        ids_keep = ids_shuffle[:num_visible]    # Indices of visible tokens (same for all batch items)
        # ids_restore = torch.argsort(ids_shuffle) # Not directly needed if using boolean mask

        # --- Create Input with Masked Areas Zeroed Out ---
        # Convert token mask (B, Fp*Tp) to pixel mask (B, F_bins, T_bins)
        pixel_mask = token_mask_to_pixel_mask(mask_tok, self.Fp, self.Tp, self.f_stride, self.t_stride) # (B, F_bins, T_bins)
        # Clone original input and apply pixel mask (zeroing out)
        masked_spec_input = spec_input.clone()
        masked_spec_input[pixel_mask.unsqueeze(1)] = 0.0 # Use unsqueezed mask for channel dim

        # --- MAE Encoder Forward Pass (Only Visible Tokens) ---
        with torch.cuda.amp.autocast(enabled=AMP):
            # Encoder receives input where masked patches are zeroed
            encoded_all, _, _ = self.enc(masked_spec_input) # (B, Fp*Tp, D_enc)
            D_enc = encoded_all.shape[-1]

            # Select only VISIBLE encoded tokens using indices
            ids_keep_batch = ids_keep.unsqueeze(0).expand(B, -1) # (B, num_visible)
            encoded_vis = torch.gather(encoded_all, dim=1, index=ids_keep_batch.unsqueeze(-1).expand(-1, -1, D_enc))
            # encoded_vis shape: (B, num_visible, D_enc)

            # -------- Decoder predicts masked pixels --------------------
            # Pass the boolean mask `mask_tok` (True where masked)
            pred_pixels_flat = self.dec(encoded_vis, mask_tok, (Fp, Tp))
            # pred_pixels_flat shape: (B * num_masked, num_pixels_per_patch)

            # -------- Get target PIXELS for masked tokens (from ORIGINAL spec) -------------
            # Extract patches from the original (dataloader-normalized) spec_input
            spec_patches = spec_input.unfold(2, self.f_stride, self.f_stride).unfold(3, self.t_stride, self.t_stride)
            # spec_patches shape: (B, 1, Fp, Tp, f_stride, t_stride)
            spec_patches = spec_patches.permute(0, 2, 3, 1, 4, 5).reshape(B, Fp * Tp, -1)
            # spec_patches shape: (B, Fp*Tp, num_pixels_per_patch)

            # Select the target patches using the boolean mask `mask_tok`
            target_pixels_flat = spec_patches[mask_tok]
            # target_pixels_flat shape: (B * num_masked, num_pixels_per_patch)

            # -------- Loss calculation ------------------------------------
            loss = self.crit(pred_pixels_flat, target_pixels_flat)

        # --- Backpropagation ---
        gnorm_total = 0.0 # Initialize gradient norm
        if is_train:
            self.opt.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            grad_norm_enc = torch.nn.utils.clip_grad_norm_(self.enc.parameters(), self.args.clip_grad)
            grad_norm_dec = torch.nn.utils.clip_grad_norm_(self.dec.parameters(), self.args.clip_grad)
            gnorm_total = float(grad_norm_enc + grad_norm_dec)
            self.scaler.step(self.opt)
            # Avoid race condition? Wait for step to finish before updating scaler
            # self.opt.synchronize() # Uncomment if using DDP FSDP? Check docs.
            self.scaler.update()

        return float(loss.item()), gnorm_total # Removed viz_data return

    # -----------------------------------------------------------------
    @torch.no_grad()
    def _eval(self):
        """ Evaluate the model on the validation set """
        self.enc.eval()
        self.dec.eval()
        avg_loss = float('inf') # Default value
        viz_data = None         # Initialize viz_data for validation
        start_time = time.time() # DEBUG TIMING

        try:
            spec_batch, *_ = next(iter(self.val_dl))   # just ONE batch
            spec_batch = spec_batch.to(self.dev) # Move to GPU
            B, F_bins, T_bins = spec_batch.shape
            spec_input = spec_batch.unsqueeze(1) # Add channel dim

            if B == 0:
                 print("--- Eval Warning: Validation batch is empty. ---")
                 return float('inf'), None # Return None for viz_data

            # This is where the autocast context starts
            with torch.cuda.amp.autocast(enabled=AMP):
                # --- Generate Mask (same logic as _step) ---
                Fp, Tp = self.Fp, self.Tp
                num_tokens = Fp * Tp
                single_mask_tok_flat = self._generate_batch_mask(Fp, Tp, self.args.mask_ratio, device=self.dev)
                mask_tok = single_mask_tok_flat.unsqueeze(0).expand(B, -1)
                num_masked = single_mask_tok_flat.sum().item()
                num_visible = num_tokens - num_masked
                ids_shuffle = torch.argsort(torch.rand(num_tokens, device=self.dev))
                ids_keep = ids_shuffle[:num_visible]
                ids_keep_batch = ids_keep.unsqueeze(0).expand(B, -1)

                # --- Encoder Forward ---
                encoded_all, _, _ = self.enc(spec_input.to(torch.half)) # Cast input here
                D_enc = encoded_all.shape[-1]

                # --- Select Visible Tokens ---
                encoded_vis = torch.gather(encoded_all, dim=1, index=ids_keep_batch.unsqueeze(-1).expand(-1, -1, D_enc))

                # --- Decoder Forward ---
                pred_pixels_flat = self.dec(encoded_vis, mask_tok, (Fp, Tp))

                # --- Target Extraction ---
                spec_patches = spec_input.unfold(2, self.f_stride, self.f_stride).unfold(3, self.t_stride, self.t_stride)
                spec_patches = spec_patches.permute(0, 2, 3, 1, 4, 5).reshape(B, Fp * Tp, -1)
                target_pixels_flat = spec_patches[mask_tok]

                # --- Loss Calculation ---
                # Ensure prediction and target are float32 for stable loss calculation
                loss = self.crit(pred_pixels_flat.float(), target_pixels_flat.float())
                avg_loss = loss.item()

                # --- Visualization Data Prep (from validation batch) ---
                if B > 0: # Prepare visualization data if batch is not empty
                    # Use the first sample and the common mask for visualization
                    spec_orig_0 = spec_batch[0].cpu() # Original spec for sample 0
                    mask_tok_0 = mask_tok[0].cpu()    # Token mask for sample 0 (same as others)

                    # Reshape predictions and targets for the first sample
                    # Predictions are ordered B*M, so take the first M predictions
                    pred_pixels_sample0 = pred_pixels_flat[:num_masked].detach().cpu()
                    # Targets are also ordered B*M
                    target_pixels_sample0 = target_pixels_flat[:num_masked].detach().cpu()

                    viz_data = {
                        "spec_orig": spec_orig_0,
                        "spec_norm": spec_orig_0, # Pass original again
                        "mask_tok": mask_tok_0,
                        "pred_pixels_flat": pred_pixels_sample0,
                        "target_pixels_flat": target_pixels_sample0,
                        "grid_shape": (Fp, Tp),
                        "patch_shape": (self.f_stride, self.t_stride)
                    }

        except StopIteration:
            print("--- Eval Warning: Validation DataLoader is empty. ---") # DEBUG
            avg_loss = float('inf') # Or handle as appropriate
        except Exception as e:
             print(f"--- Eval Error: An exception occurred during validation: {e} ---") # DEBUG
             import traceback
             traceback.print_exc()
             avg_loss = float('inf') # Indicate error

        eval_time = time.time() - start_time
        print(f"Validation (1 batch) finished in {eval_time:.2f}s")

        self.enc.train() # Set back to train mode
        self.dec.train()
        return avg_loss, viz_data # Return loss AND visualization data

    # -----------------------------------------------------------------
    def train(self):
        """ Main training loop """
        start_time = time.time()
        print(f"Starting training from step {self.step}")
        dl_iter = iter(self.train_dl) # Initialize iterator outside loop

        while self.step < self.args.steps:
            # --- Warmup Phase ---
            if self.step < self.args.warmup_steps:
                 lr_scale = min(1.0, float(self.step + 1) / self.args.warmup_steps)
                 current_lr_target = self.args.lr * lr_scale
            # --- Post-Warmup / Cosine Decay Phase ---
            else:
                 # Calculate cosine decay progress
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
            train_loss, gnorm_total = self._step(spec_batch, is_train=True) # No longer returns viz_data

            # --- Logging & Evaluation ---
            if self.step % self.args.eval_interval == 0 or self.step == self.args.steps: # Log/Eval on interval or last step
                val_loss, val_viz_data = self._eval()

                # EMA Loss Calculation
                if self.train_loss_ema is None:
                    self.train_loss_ema = train_loss
                    self.val_loss_ema = val_loss if not math.isnan(val_loss) else train_loss # Handle potential initial NaN val_loss
                else:
                    self.train_loss_ema = self.alpha * train_loss + (1 - self.alpha) * self.train_loss_ema
                    if not math.isnan(val_loss):
                         self.val_loss_ema = self.alpha * val_loss + (1 - self.alpha) * self.val_loss_ema

                current_lr = self.opt.param_groups[0]['lr'] # Get actual LR after potential warmup/decay
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
                       # Use try-except for val_loss formatting in case it's NaN/inf
                       f"val_loss {'%.4f'%val_loss if isinstance(val_loss, float) and not math.isnan(val_loss) and not math.isinf(val_loss) else str(val_loss)} (ema {'%.4f'%self.val_loss_ema if self.val_loss_ema is not None else 'N/A'}) | "
                       f"lr {current_lr:.2e} | gnorm {gnorm_total:.2e} | "
                       f"{steps_per_sec:.1f} steps/s | eta {eta_str}")
                self.log(msg)

                # --- Visualization ---
                if val_viz_data: # Use visualization data from the validation step
                    _viz_triptych(
                        self.run_dir/'imgs', f"step_{self.step:07d}",
                        viz_data=val_viz_data # Pass the validation data
                    ) # Consider renaming viz file to 'viz_val_...' if needed

                # --- Checkpointing ---
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.bad_evals = 0
                    print(f"  New best val_loss: {self.best_val_loss:.4f}")
                    # Save best checkpoint only
                    best_path = self.run_dir/'weights'/'best.pt'
                    payload = {
                        'step': self.step,
                        'enc': self.enc.state_dict(),
                        'dec': self.dec.state_dict(),
                        'opt': self.opt.state_dict(),
                        'sched': self.sched.state_dict(), # Save scheduler state
                        'scaler': self.scaler.state_dict(),
                        'best_val_loss': self.best_val_loss,
                        'train_loss_ema': self.train_loss_ema,
                        'val_loss_ema': self.val_loss_ema,
                        'hist': self.hist,
                        'args': vars(self.args) # Save args used for this run
                    }
                    torch.save(payload, best_path)
                    print(f"  Saved best checkpoint to {best_path}")
                else:
                    self.bad_evals += 1
                # Save checkpoint at every eval interval as step_{step}.pt
                step_ckpt_path = self.run_dir/'weights'/f'step_{self.step:07d}.pt'
                torch.save(payload, step_ckpt_path)

                # Early stopping check
                if self.args.early_stop > 0 and self.bad_evals >= self.args.early_stop:
                    self.log(f"Early stopping triggered after {self.args.early_stop} evaluations without improvement.")
                    break

                start_time = time.time() # Reset timer for next interval


        # --- Final Actions ---
        self.log("Training finished.")
        # Final plot
        df = pd.DataFrame(self.hist)
        df.to_csv(self.run_dir/'metrics.csv', index=False)
        try:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(df.step, df.train_loss, label='Train Loss', alpha=0.7)
            plt.plot(df.step, df.val_loss, label='Val Loss', alpha=0.7)
            # Add EMA plots if available
            if 'train_loss_ema' in df.columns:
                 plt.plot(df.step, df.train_loss_ema, label='Train Loss EMA', linestyle='--')
            if 'val_loss_ema' in df.columns:
                 plt.plot(df.step, df.val_loss_ema, label='Val Loss EMA', linestyle='--')

            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss Curve')
            plt.grid(True, alpha=0.5)
            plt.ylim(bottom=0) # Loss should not be negative

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

        self.log.close() # Close log file


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
        # Use a distinct value for masking, e.g., slightly below min or 0 if appropriate
        mask_fill_value = spec_orig.min() - (spec_orig.max() - spec_orig.min()) * 0.1
        spec_masked_viz[pixel_mask] = mask_fill_value

        # --- Reconstruct Spectrogram Image ---
        recon_spec = spec_orig.clone() # Start with original visible patches

        if num_masked > 0:
            # Reshape predicted pixels back to patches: (num_masked, f_stride, t_stride)
            pred_patches = pred_pixels_flat.reshape(num_masked, 1, f_stride, t_stride)

            # Get indices of masked tokens
            masked_indices_flat = torch.nonzero(mask_tok).squeeze(-1)
            rows, cols = np.unravel_index(masked_indices_flat.numpy(), (Fp, Tp))

            # Scatter predicted patches into the recon_spec
            for i, (r, c) in enumerate(zip(rows, cols)):
                f_start, f_end = r * f_stride, (r + 1) * f_stride
                t_start, t_end = c * t_stride, (c + 1) * t_stride
                recon_spec[f_start:f_end, t_start:t_end] = pred_patches[i, 0] # Remove channel dim

        # --- Calculate Pixel MSE Map (Optional) ---
        mse_map = torch.zeros_like(spec_orig)
        if num_masked > 0:
            # Also need target pixels for MSE map
            target_pixels_flat = viz_data["target_pixels_flat"] # (num_masked, P*P)
            target_patches = target_pixels_flat.reshape(num_masked, 1, f_stride, t_stride)
            mse_per_patch = tF.mse_loss(pred_patches, target_patches, reduction='none').mean(dim=(1, 2, 3))

            for i, (r, c) in enumerate(zip(rows, cols)):
                f_start, f_end = r * f_stride, (r + 1) * f_stride
                t_start, t_end = c * t_stride, (c + 1) * t_stride
                mse_map[f_start:f_end, t_start:t_end] = mse_per_patch[i]
        mse_map[~pixel_mask] = 0.0 # Zero out visible regions

        # --- FIX: Define vmin and vmax ---
        # Determine robust color limits for spectrograms based on original input
        # Ensure spec_orig is float for quantile calculation
        spec_orig_float = spec_orig.float()
        vmin = torch.quantile(spec_orig_float, 0.01)
        vmax = torch.quantile(spec_orig_float, 0.99)
        # Handle potential edge case where min/max are equal
        if vmin == vmax:
            vmin = spec_orig_float.min()
            vmax = spec_orig_float.max()
            if vmin == vmax: # If still equal (e.g., all zeros)
                vmax = vmin + 1e-6 # Add small epsilon to avoid division by zero in color mapping

        # ---------- Plotting ----------------------------------------------------
        fig, ax = plt.subplots(2, 3, figsize=(15, 8), dpi=120)
        kw_spec = dict(aspect='auto', origin='lower', cmap='magma')
        kw_err = dict(aspect='auto', origin='lower', cmap='coolwarm', interpolation='nearest')
        kw_diff = dict(aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')

        # Row 1
        ax[0,0].imshow(spec_orig, **kw_spec, vmin=vmin, vmax=vmax)
        ax[0,0].set_title('Input Spectrogram')
        ax[0,1].imshow(spec_masked_viz, **kw_spec, vmin=vmin, vmax=vmax)
        ax[0,1].set_title('Masked Input')
        im_err = ax[0,2].imshow(mse_map, **kw_err)
        ax[0,2].set_title('Pixel MSE in Masked Areas')
        fig.colorbar(im_err, ax=ax[0,2], fraction=0.046, pad=0.04)

        # Row 2
        # --- Bottom Left: Mask Boolean ---
        # Display the pixel_mask (True=masked=white, False=visible=black)
        # Ensure pixel_mask is on CPU and float for imshow
        ax[1,0].imshow(pixel_mask.cpu().float(), cmap='gray', origin='lower', aspect='auto', vmin=0, vmax=1)
        ax[1,0].set_title('Mask Boolean (True=Masked)')
        # --- Bottom Middle: Reconstructed Spectrogram ---
        im_recon = ax[1,1].imshow(recon_spec, **kw_spec, vmin=vmin, vmax=vmax)
        ax[1,1].set_title('Reconstructed Spectrogram')
        fig.colorbar(im_recon, ax=ax[1,1], fraction=0.046, pad=0.04)
        # --- Bottom Right: Absolute Difference ---
        diff_map = (spec_orig - recon_spec).abs()
        im_diff = ax[1,2].imshow(diff_map, **kw_diff)
        ax[1,2].set_title('Absolute Difference')
        fig.colorbar(im_diff, ax=ax[1,2], fraction=0.046, pad=0.04)

        # General settings
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
            plt.close(fig) # Attempt to close figure even if error occurred


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
    # Note: Learning rate might need scaling based on batch size, e.g., lr = base_lr * total_bs / 256
    p.add_argument('--lr',       type=float, default=5e-4, help="Base learning rate (before warmup/decay)")
    p.add_argument('--lr_min',   type=float, default=1e-5, help="Minimum learning rate for cosine decay")
    p.add_argument('--wd',       type=float, default=0.05, help="Weight decay")
    p.add_argument('--mask_ratio', type=float, default=0.75, help="Fraction of patches to mask")
    p.add_argument('--clip_grad', type=float, default=1.0, help="Gradient clipping value")

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
