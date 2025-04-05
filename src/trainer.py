# trainer.py
import os
import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import json
from torch.cuda.amp import autocast, GradScaler
from model import BirdJEPA, EMA
from data_class import BirdJEPA_Dataset, collate_fn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import shutil
from datetime import datetime
import math
from utils import load_model
import sys
import io
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from timing_utils import Timer, timed_operation, timing_stats

class LogCapture:
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, "w")

    def write(self, message):
        # Write to log file only
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal

class ModelTrainer:
    def __init__(
        self, 
        model, 
        train_loader, 
        test_loader, 
        device, 
        max_steps=1000, 
        eval_interval=100, 
        save_interval=500,
        weights_save_dir="saved_weights", 
        experiment_dir="experiments",
        trailing_avg_window=100, 
        early_stopping=True, 
        patience=4,
        ema_momentum=0.9,
        verbose=False, 
        overfit_single_batch=False,
        encoder_lr=1e-4,
        predictor_lr=1e-3,
        decoder_lr=1e-4,
        freeze_encoder_steps=0,
        freeze_decoder_steps=0,
        debug=False,
        lr=None   # optional single-lr param from older code
    ):
        self.model = model
        self.model.ema_m = ema_momentum
        self.model.debug = debug
        self.train_iter = train_loader
        self.test_iter = test_loader
        self.device = device
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.weights_save_dir = weights_save_dir
        self.experiment_dir = experiment_dir
        self.trailing_avg_window = trailing_avg_window
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose
        self.overfit_single_batch = overfit_single_batch
        self.debug = debug
        
        # Initialize timing_stats with the experiment directory
        timing_stats.set_experiment_name(os.path.basename(experiment_dir))
        timing_stats.log_dir = experiment_dir
        
        # Set up log capturing if debug mode is enabled
        if self.debug:
            log_file_path = os.path.join(self.experiment_dir, 'debug_log.txt')
            # Only display critical debug info to terminal, full logs go to file
            self.log_capture = LogCapture(log_file_path)
            sys.stdout = self.log_capture
            # Write a message to the terminal before redirecting
            print(f"Debug logging enabled - output will be saved to: {log_file_path}")

        # if user provided old single-lr, override
        if lr is not None:
            encoder_lr = lr
            predictor_lr = lr
            decoder_lr = lr

        self.encoder_lr = encoder_lr
        self.predictor_lr = predictor_lr
        self.decoder_lr = decoder_lr
        self.freeze_encoder_steps = freeze_encoder_steps
        self.freeze_decoder_steps = freeze_decoder_steps

        # set up separate optimizers
        self.encoder_optimizer = torch.optim.AdamW(
            self.model.context_encoder.parameters(),
            lr=self.encoder_lr,
            weight_decay=0.0
        )
        self.predictor_optimizer = torch.optim.AdamW(
            self.model.predictor.parameters(),
            lr=self.predictor_lr,
            weight_decay=0.0
        )
        self.decoder_optimizer = torch.optim.AdamW(
            self.model.decoder.parameters(),
            lr=self.decoder_lr,
            weight_decay=0.0
        )

        # print param counts
        context_params = sum(p.numel() for p in self.model.context_encoder.parameters())
        target_params  = sum(p.numel() for p in self.model.target_encoder.parameters())
        predictor_params = sum(p.numel() for p in self.model.predictor.parameters())
        decoder_params   = sum(p.numel() for p in self.model.decoder.parameters())
        total_params     = sum(p.numel() for p in self.model.parameters())

        print("\nModel Parameter Counts:")
        print(f"Context Encoder:  {context_params:,} parameters")
        print(f"Target Encoder:   {target_params:,} parameters")
        print(f"Predictor:        {predictor_params:,} parameters")
        print(f"Decoder:          {decoder_params:,} parameters")
        print(f"Total:            {total_params:,} parameters\n")

        # directories
        os.makedirs(self.weights_save_dir, exist_ok=True)
        self.predictions_subfolder_path = os.path.join(self.experiment_dir, 'predictions')
        os.makedirs(self.predictions_subfolder_path, exist_ok=True)

        # store embedding stats for plotting
        self.embedding_stats_history = {
            'context_variance': [], 'context_distance': [], 'context_norm': [],
            'target_variance': [], 'target_distance': [], 'target_norm': [],
            'pred_variance': [],   'pred_distance': [],   'pred_norm': []
        }

        # early stopping info
        if self.early_stopping:
            print(f"\nEarly Stopping Configuration:")
            print(f"Patience: {self.patience} validation intervals")
            print(f"Will stop after {self.patience * self.eval_interval} steps without improvement")

        print(f"\nEMA Configuration:")
        print(f"Initial momentum: {ema_momentum}")
        print("Final momentum: 1.0")

    def embedding_variance(self, embeddings):
        return embeddings.var().item()

    def print_to_terminal(self, message):
        """Print a message to both the log file and the terminal."""
        if hasattr(self, 'log_capture'):
            # Write to log file
            self.log_capture.write(message + "\n")
            # Also print to the original terminal
            if hasattr(self.log_capture, 'terminal'):
                self.log_capture.terminal.write(message + "\n")
                self.log_capture.terminal.flush()
        else:
            # If no log capture, just print normally
            print(message)

    def save_model(self, step, training_stats, raw_loss_list=None, raw_val_loss_list=None):
        # create checkpoint
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'encoder_optimizer_state': self.encoder_optimizer.state_dict(),
            'predictor_optimizer_state': self.predictor_optimizer.state_dict(),
            'decoder_optimizer_state': self.decoder_optimizer.state_dict(),
        }

        # store training stats in the checkpoint so we can continue training
        checkpoint['training_stats'] = training_stats
        checkpoint['raw_loss_list'] = raw_loss_list if raw_loss_list is not None else []
        checkpoint['raw_val_loss_list'] = raw_val_loss_list if raw_val_loss_list is not None else []

        # save
        ckpt_path = os.path.join(self.weights_save_dir, f'checkpoint_{step}.pt')
        torch.save(checkpoint, ckpt_path)

        # store embedding stats
        training_stats['embedding_stats_history'] = self.embedding_stats_history

        # convert to json-safe format
        json_safe_stats = {}
        for key, val in training_stats.items():
            if isinstance(val, (list, tuple)) and len(val) and torch.is_tensor(val[0]):
                # list of tensors
                json_safe_stats[key] = [float(x.cpu().detach()) for x in val]
            elif torch.is_tensor(val):
                json_safe_stats[key] = float(val.cpu().detach())
            elif key == 'embedding_stats_history':
                # nested dict of lists
                nested = {}
                for k_, v_ in val.items():
                    nested[k_] = [float(x) for x in v_]
                json_safe_stats[key] = nested
            else:
                json_safe_stats[key] = val

        stats_file = os.path.join(self.experiment_dir, "training_statistics.json")
        with open(stats_file, 'w') as jf:
            json.dump(json_safe_stats, jf)

        # Make sure any captured logs are flushed to disk
        if self.debug and hasattr(self, 'log_capture'):
            self.log_capture.flush()

    def load_model(self, checkpoint_path=None, checkpoint=None):
        """
        load model and optimizer states from checkpoint
        args:
            checkpoint_path: path to checkpoint file (optional)
            checkpoint: checkpoint dictionary (optional)
            
        one of checkpoint_path or checkpoint must be provided
        """
        if checkpoint is None and checkpoint_path is None:
            raise ValueError("Either checkpoint_path or checkpoint must be provided")
        
        if checkpoint is None:
            checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state'])
        self.predictor_optimizer.load_state_dict(checkpoint['predictor_optimizer_state'])
        self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state'])

        # new logic for safely loading training stats
        training_stats = checkpoint.get('training_stats', {})
        raw_loss_list = checkpoint.get('raw_loss_list', [])
        raw_val_loss_list = checkpoint.get('raw_val_loss_list', [])
        last_step = checkpoint.get('step', 0)

        return last_step, training_stats, raw_loss_list, raw_val_loss_list

    @timed_operation("train_epoch")
    def train_epoch(self, epoch):
        with Timer("train_epoch_wrapper", debug=self.debug):
            self.model.train()
            total_loss = 0
            num_batches = len(self.train_iter)

            with Timer("epoch_initialization", debug=self.debug):
                self.print_to_terminal(f"\nEpoch {epoch}\n")
                self.print_to_terminal("Training...\n")

            for batch_idx, batch in enumerate(self.train_iter):
                with Timer("batch_processing", debug=self.debug):
                    # Move data to device
                    with Timer("data_transfer", debug=self.debug):
                        full_spect, targ_spect, cont_spect, labels, mask, file_names = batch
                        cont_spect = cont_spect.to(self.device)
                        targ_spect = targ_spect.to(self.device)
                        mask       = mask.to(self.device)

                    # Forward pass
                    with Timer("forward_pass", debug=self.debug):
                        loss, diff, pred, target_repr, context_repr = self.model.compute_latent_loss(
                            cont_spect, targ_spect, mask
                        )

                    # Backward pass
                    with Timer("backward_pass", debug=self.debug):
                        self.encoder_optimizer.zero_grad(set_to_none=True)
                        self.predictor_optimizer.zero_grad(set_to_none=True)
                        self.decoder_optimizer.zero_grad(set_to_none=True)
                        scaler = GradScaler()
                        scaler.scale(loss).backward()

                    if batch_idx >= self.freeze_encoder_steps and has_gradients(self.encoder_optimizer):
                        scaler.unscale_(self.encoder_optimizer)
                        scaler.step(self.encoder_optimizer)

                    if batch_idx >= self.freeze_decoder_steps and has_gradients(self.decoder_optimizer):
                        scaler.unscale_(self.decoder_optimizer)
                        scaler.step(self.decoder_optimizer)

                    if has_gradients(self.predictor_optimizer):
                        scaler.unscale_(self.predictor_optimizer)
                        scaler.step(self.predictor_optimizer)

                    scaler.update()

                    total_loss += loss.item()

                    # Update EMA
                    with Timer("ema_update", debug=self.debug):
                        self.model.update_ema()

                    # Log progress
                    if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                        with Timer("progress_logging", debug=self.debug):
                            avg_loss = total_loss / (batch_idx + 1)
                            self.print_to_terminal(f"Batch {batch_idx}/{num_batches}, Loss: {avg_loss:.4f}\n")

            # Calculate epoch statistics
            with Timer("epoch_statistics", debug=self.debug):
                avg_epoch_loss = total_loss / num_batches
                self.print_to_terminal(f"Epoch {epoch} Average Loss: {avg_epoch_loss:.4f}\n")

            return avg_epoch_loss

    @timed_operation("validate")
    def validate(self, epoch):
        with Timer("validate_wrapper", debug=self.debug):
            self.model.eval()
            total_loss = 0
            num_batches = len(self.test_iter)

            with Timer("validation_initialization", debug=self.debug):
                self.print_to_terminal("\nValidation...\n")

            with torch.no_grad():
                for batch_idx, batch in enumerate(self.test_iter):
                    with Timer("validation_batch_processing", debug=self.debug):
                        # Move data to device
                        with Timer("validation_data_transfer", debug=self.debug):
                            full_spect, targ_spect, cont_spect, labels, mask, file_names = batch
                            cont_spect = cont_spect.to(self.device)
                            targ_spect = targ_spect.to(self.device)
                            mask       = mask.to(self.device)

                        # Forward pass
                        with Timer("validation_forward_pass", debug=self.debug):
                            loss, diff, pred_sequence, _, _ = self.model.compute_latent_loss(
                                cont_spect, targ_spect, mask, is_eval_step=True
                            )

                        total_loss += loss.item()

                        # Log progress
                        if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                            with Timer("validation_progress_logging", debug=self.debug):
                                avg_loss = total_loss / (batch_idx + 1)
                                self.print_to_terminal(f"Validation Batch {batch_idx}/{num_batches}, Loss: {avg_loss:.4f}\n")

            # Calculate validation statistics
            with Timer("validation_statistics", debug=self.debug):
                avg_val_loss = total_loss / num_batches
                self.print_to_terminal(f"Validation Average Loss: {avg_val_loss:.4f}\n")

            return avg_val_loss

    @timed_operation("train")
    def train(self, continue_training=False, training_stats=None, last_step=0, raw_loss_list=None, raw_val_loss_list=None):
        # Add device information at start of training
        self.print_to_terminal(f"\nTraining on device: {self.device}")
        if torch.cuda.is_available():
            self.print_to_terminal(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            self.print_to_terminal(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB allocated, {torch.cuda.memory_reserved()/1024**2:.1f}MB reserved")
        else:
            self.print_to_terminal("CUDA is not available. Using CPU only.")
        
        # Ensure the experiment directory exists
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Ensure debug log is properly set up
        if self.debug and not hasattr(self, 'log_capture'):
            log_file_path = os.path.join(self.experiment_dir, 'debug_log.txt')
            self.log_capture = LogCapture(log_file_path)
            sys.stdout = self.log_capture
            # Make sure this message appears in the terminal before redirection
            self.print_to_terminal(f"Debug logging initialized at start of training")
        
        # Counter to flush logs regularly
        flush_every = max(1, min(10, self.eval_interval // 10))  # Adjust this to control flush frequency
        
        # if no arrays were passed in, init fresh
        if raw_loss_list is None:
            raw_loss_list = []
        if raw_val_loss_list is None:
            raw_val_loss_list = []

        step = last_step + 1 if continue_training else 0
        scaler = GradScaler()

        # initialize or load training stats
        if not training_stats:
            stats = {
                'training_loss': [],
                'validation_loss': [],
                'steps_since_improvement': 0,
                'best_val_loss': float('inf'),
                'grad_stats': [],
                'network_stats': [],
                'embedding_stats': [],
                'learning_rates': []
            }
        else:
            stats = training_stats

        best_val_loss = stats.get('best_val_loss', float('inf'))
        steps_since_improvement = stats.get('steps_since_improvement', 0)

        train_iter = iter(self.train_iter)
        test_iter = iter(self.test_iter)

        def has_gradients(optimizer):
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        return True
            return False

        with Timer("training_loop", debug=self.debug):
            while step < self.max_steps:
                # Overall batch training timer
                with Timer("train_step_total", debug=self.debug):
                    # Data loading timer - TRAINING BATCH ONLY
                    with Timer("data_loading", debug=self.debug):
                        try:
                            if not self.overfit_single_batch:
                                # Only load training batch here
                                batch = next(train_iter)
                            else:
                                batch = next(train_iter)
                                val_batch = batch  # For overfit, use same batch
                        except StopIteration:
                            if not self.overfit_single_batch:
                                train_iter = iter(self.train_iter)
                                continue
                            
                        self.adjust_momentum(step)
                        
                        full_spect, targ_spect, cont_spect, labels, mask, file_names = batch
                        cont_spect = cont_spect.to(self.device)
                        targ_spect = targ_spect.to(self.device)
                        mask       = mask.to(self.device)

                    self.model.train()

                    # Zero gradients timer
                    with Timer("zero_grad", debug=self.debug):
                        self.encoder_optimizer.zero_grad(set_to_none=True)
                        self.predictor_optimizer.zero_grad(set_to_none=True)
                        self.decoder_optimizer.zero_grad(set_to_none=True)

                    # Forward pass timer
                    with Timer("forward_pass", debug=self.debug):
                        with autocast():
                            loss, diff, pred, target_repr, context_repr = self.model.compute_latent_loss(
                                cont_spect, targ_spect, mask
                            )

                    # Backward pass timer
                    with Timer("backward_pass", debug=self.debug):
                        scaler.scale(loss).backward()

                    # Optimizer step timer
                    with Timer("optimizer_step", debug=self.debug):
                        if step >= self.freeze_encoder_steps and has_gradients(self.encoder_optimizer):
                            scaler.unscale_(self.encoder_optimizer)
                            scaler.step(self.encoder_optimizer)

                        if step >= self.freeze_decoder_steps and has_gradients(self.decoder_optimizer):
                            scaler.unscale_(self.decoder_optimizer)
                            scaler.step(self.decoder_optimizer)

                        if has_gradients(self.predictor_optimizer):
                            scaler.unscale_(self.predictor_optimizer)
                            scaler.step(self.predictor_optimizer)

                    # EMA update timer
                    with Timer("ema_update", debug=self.debug):
                        scaler.update()
                        self.model.update_ema()

                # Only do validation at eval_interval steps
                if step % self.eval_interval == 0:
                    # Load validation batch ONLY at eval_interval steps
                    with Timer("validation_data_loading", debug=self.debug):
                        try:
                            if not self.overfit_single_batch:
                                # Only load validation batch when needed
                                val_batch = next(test_iter)
                            # else: val_batch already set in overfit mode
                        except StopIteration:
                            test_iter = iter(self.test_iter)
                            val_batch = next(test_iter)

                    # Validation step (outside the train_step_total timer)
                    val_full, val_target, val_cont, val_labels, val_mask, val_fnames = val_batch
                    
                    # Do full validation with visualizations
                    val_loss, _ = self.validate_model(
                        step,
                        context_spec=val_cont.to(self.device),
                        target_spec=val_target.to(self.device),
                        mask=val_mask.to(self.device),
                        vocalization=val_labels.to(self.device),
                        do_visualize=True
                    )
                    
                    # Print loss at eval_interval steps
                    step_msg = f"Step {step}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}"
                    if hasattr(self, 'log_capture') and hasattr(self.log_capture, 'terminal'):
                        self.log_capture.terminal.write(step_msg + "\n")
                        self.log_capture.terminal.flush()
                    else:
                        print(step_msg)
                    
                    # Add validation loss to records
                    raw_val_loss_list.append(val_loss)
                else:
                    # For non-eval steps, just use the last known validation loss
                    val_loss = raw_val_loss_list[-1] if raw_val_loss_list else float('inf')
                
                # Always record training loss
                raw_loss_list.append(loss.item())
                
                # Record losses in stats for continuity
                stats['training_loss'].append(loss.item())
                stats['validation_loss'].append(val_loss)

                if step % self.eval_interval == 0:
                    if step == 0:
                        smoothed_training_loss = loss.item()
                        smoothed_val_loss = val_loss
                    else:
                        def ema(vals, w):
                            alpha = 2/(w+1)
                            out = [vals[0]]
                            for x in vals[1:]:
                                out.append(alpha*x + (1-alpha)*out[-1])
                            return out
                        train_recent = raw_loss_list[-self.trailing_avg_window:]
                        val_recent   = raw_val_loss_list[-self.trailing_avg_window:]

                        s_train = ema(train_recent, self.trailing_avg_window)
                        s_val   = ema(val_recent,   self.trailing_avg_window)

                        smoothed_training_loss = s_train[-1] if s_train else loss.item()
                        smoothed_val_loss      = s_val[-1]   if s_val   else val_loss

                    enc_lr = self.encoder_optimizer.param_groups[0]['lr']
                    dec_lr = self.decoder_optimizer.param_groups[0]['lr']
                    pred_lr= self.predictor_optimizer.param_groups[0]['lr']

                    c_stats = self.get_network_stats(self.model.context_encoder, "context")
                    gnorm   = c_stats["context_grad_norm"]

                    with torch.no_grad():
                        c_repr, _ = self.model.context_encoder(cont_spect)
                        t_repr, _ = self.model.target_encoder(targ_spect)
                        c_var = self.embedding_variance(c_repr)
                        t_var = self.embedding_variance(t_repr)

                    # Use our new method to print progress to terminal
                    progress_msg = f"\nStep [{step}/{self.max_steps}] - " \
                                   f"Smoothed Train Loss: {smoothed_training_loss:.4f}, " \
                                   f"Smoothed Val Loss: {smoothed_val_loss:.4f}, " \
                                   f"Vars [C/T]: {c_var:.4f}/{t_var:.4f} - " \
                                   f"GradNorm: {gnorm:.2e} - " \
                                   f"LRs: enc={enc_lr:.2e}, pred={pred_lr:.2e}, dec={dec_lr:.2e}"
                    self.print_to_terminal(progress_msg)
                    # Also log it to the file
                    print(progress_msg)

                    if smoothed_val_loss < best_val_loss:
                        best_val_loss = smoothed_val_loss
                        steps_since_improvement = 0
                    else:
                        steps_since_improvement += 1

                    if self.early_stopping and steps_since_improvement >= self.patience:
                        early_stop_msg = f"\nEarly stopping triggered at step {step}. no improvement for {self.patience} intervals."
                        self.print_to_terminal(early_stop_msg)
                        print(early_stop_msg)  # Also log to file
                        # save final checkpoint before stopping
                        stats['best_val_loss'] = best_val_loss
                        stats['steps_since_improvement'] = steps_since_improvement
                        self.save_model(step, stats, raw_loss_list, raw_val_loss_list)
                        break

                    c_stats = self.get_network_stats(self.model.context_encoder, "context")
                    lr_stats = {
                        'encoder_lr': self.encoder_optimizer.param_groups[0]['lr'],
                        'predictor_lr': self.predictor_optimizer.param_groups[0]['lr'],
                        'decoder_lr': self.decoder_optimizer.param_groups[0]['lr']
                    }
                    emb_stats = {
                        'context_var': c_var,
                        'target_var': t_var,
                        'grad_norm': gnorm
                    }
                    stats['grad_stats'].append(c_stats)
                    stats['learning_rates'].append(lr_stats)
                    stats['embedding_stats'].append(emb_stats)

                if step % self.save_interval == 0:
                    stats['best_val_loss'] = best_val_loss
                    stats['steps_since_improvement'] = steps_since_improvement
                    self.save_model(step, stats, raw_loss_list, raw_val_loss_list)

                step += 1
                
                # Flush the logs periodically to ensure they are saved
                if self.debug and hasattr(self, 'log_capture'):
                    if step % flush_every == 0:  # Flush more frequently than eval_interval
                        self.log_capture.flush()
                        if step % (flush_every * 10) == 0:  # Occasionally log progress to file
                            print(f"[Training step {step}/{self.max_steps}] in progress...")

        # done loop
        final_stats = {
            'training_loss': stats['training_loss'],
            'validation_loss': stats['validation_loss'],
            'best_val_loss': best_val_loss,
            'steps_since_improvement': steps_since_improvement
        }
        # merge final with existing stats
        stats.update(final_stats)
        self.save_model(step-1, stats, raw_loss_list, raw_val_loss_list)
        
        # Save timing statistics
        timing_stats.save_stats()
        
        return raw_loss_list, raw_val_loss_list

    def validate_model(self, step, context_spec, target_spec, mask, vocalization, do_visualize=False):
        with Timer("validation_model", debug=self.debug):
            self.model.eval()
            if self.debug and hasattr(self, 'log_capture'):
                self.log_capture.write("\n[VALIDATION] Beginning validation forward pass\n")
                self.log_capture.flush()
                
                # Dump data loader timing stats to debug log
                try:
                    import logging
                    logger = logging.getLogger('data_class')
                    
                    # Get dataset objects from data loaders
                    train_dataset = self.train_iter.dataset
                    test_dataset = self.test_iter.dataset
                    
                    logger.info(f"\n===== DETAILED DATA LOADER TIMING AT STEP {step} =====")
                    
                    # Log reservoir information
                    if hasattr(train_dataset, 'reservoir'):
                        reservoir_size = len(train_dataset.reservoir)
                        reservoir_bytes = train_dataset.reservoir_bytes
                        target_bytes = train_dataset.target_reservoir_bytes
                        logger.info(f"Reservoir status: {reservoir_size} files, " 
                                   f"{reservoir_bytes/1024**3:.2f}/{target_bytes/1024**3:.2f} GB used")
                    
                    # Log cache performance
                    if hasattr(train_dataset, 'reservoir_hits'):
                        logger.info(f"Cache performance: "
                                   f"reservoir hits={train_dataset.reservoir_hits}, "
                                   f"misses={train_dataset.reservoir_misses}, "
                                   f"LRU hits={train_dataset.cache_hits}, "
                                   f"misses={train_dataset.cache_misses}")
                    
                    # Log timing statistics
                    if hasattr(train_dataset, 'timing_data'):
                        logger.info("Timing statistics:")
                        for key, times in train_dataset.timing_data.items():
                            if times:
                                avg = sum(times) / len(times)
                                max_time = max(times)
                                min_time = min(times)
                                logger.info(f"  {key}: avg={avg:.6f}s, min={min_time:.6f}s, "
                                           f"max={max_time:.6f}s, samples={len(times)}")
                
                    # Get timing statistics from timing_utils if available
                    try:
                        from timing_utils import timing_stats
                        logger.info("\nGlobal timing statistics:")
                        for op_name, stats in timing_stats.items():
                            if stats['count'] > 0:
                                avg = stats['total_time'] / stats['count']
                                logger.info(f"  {op_name}: {avg:.6f}s avg, {stats['total_time']:.2f}s total, "
                                          f"{stats['count']} calls")
                    except (ImportError, AttributeError) as e:
                        logger.warning(f"Could not get timing stats: {e}")
                        
                    logger.info("="*50)
                
                except Exception as e:
                    print(f"Error dumping timing info: {e}")
            
            with torch.no_grad():
                # Forward pass with timing
                with Timer("forward_pass", debug=self.debug):
                    # All forward propagation steps in a single block
                    context_repr, context_intermediate = self.model.context_encoder(context_spec)
                    target_repr, target_intermediate = self.model.target_encoder(target_spec)
                    loss, diff, pred_sequence, _, _ = self.model.compute_latent_loss(
                        context_spec, target_spec, mask, is_eval_step=True
                    )
                
                # Always compute embedding stats for monitoring
                c_stats = self.compute_embedding_stats(context_repr)
                t_stats = self.compute_embedding_stats(target_repr)
                p_stats = self.compute_embedding_stats(pred_sequence)
                self.update_embedding_stats(context_repr, target_repr, pred_sequence)

                # Only do visualizations and detailed stats at eval_interval
                if do_visualize:
                    with Timer("validation_stats_computation", debug=self.debug):
                        embed_stats_msg = "\nEmbedding Statistics:"
                        embed_stats_msg += f"\nContext var={c_stats['embedding_variance']:.4f}, dist={c_stats['avg_pairwise_distance']:.4f}, norm={c_stats['avg_embedding_norm']:.4f}"
                        embed_stats_msg += f"\nTarget  var={t_stats['embedding_variance']:.4f}, dist={t_stats['avg_pairwise_distance']:.4f}, norm={t_stats['avg_embedding_norm']:.4f}"
                        embed_stats_msg += f"\nPred    var={p_stats['embedding_variance']:.4f}, dist={p_stats['avg_pairwise_distance']:.4f}, norm={p_stats['avg_embedding_norm']:.4f}"
                        
                        # Print to both log file and terminal
                        self.print_to_terminal(embed_stats_msg)

                    with Timer("latent_predictions_visualization", debug=self.debug):
                        self.visualize_latent_predictions(
                            step=step,
                            context_spectrogram=context_spec,
                            target_spectrogram=target_spec,
                            context_repr=context_repr,
                            pred_sequence=pred_sequence,
                            target_repr=target_repr,
                            mask=mask
                        )
                    
                    with Timer("stacked_layers_visualization", debug=self.debug):
                        self.visualize_stacked_layers(
                            step=step,
                            context_spectrogram=context_spec,
                            target_spectrogram=target_spec,
                            context_intermediate=context_intermediate,
                            target_intermediate=target_intermediate,
                            mask=mask
                        )
                
                # Flush the log after the whole validation step
                if hasattr(self, 'log_capture'):
                    self.log_capture.flush()

                return loss.item(), {}

    @timed_operation("visualize_latent_predictions")
    def visualize_latent_predictions(
        self, 
        step,
        context_spectrogram,
        target_spectrogram,
        context_repr,
        pred_sequence,
        target_repr,
        mask
    ):
        # Move all tensors to CPU and convert to numpy at once
        with Timer("latent_predictions_data_preparation", debug=self.debug):
            context_spec_np = context_spectrogram[0].cpu().numpy()
            target_spec_np = target_spectrogram[0].cpu().numpy()
            mask_np = mask[0].cpu().numpy()
            context_repr_np = context_repr[0].cpu().numpy().T
            target_repr_np = target_repr[0].cpu().numpy().T
            pred_sequence_np = pred_sequence[0].cpu().numpy().T
            mask_expanded_np = mask.unsqueeze(-1)[0].cpu().numpy()
            lat_error = ((pred_sequence - target_repr)**2 * mask.unsqueeze(-1))[0].cpu().numpy().T

        with Timer("latent_predictions_figure_creation", debug=self.debug):
            fig, axs = plt.subplots(2, 3, figsize=(20,10))
            axs = axs.flatten()

        with Timer("latent_predictions_plotting", debug=self.debug):
            # Plot all at once without individual timing
            axs[0].imshow(context_spec_np, aspect='auto', origin='lower')
            axs[0].set_title(f"context spectrogram (step {step})")
            self._add_mask_overlay(axs[0], torch.from_numpy(mask_np))

            axs[1].imshow(target_spec_np, aspect='auto', origin='lower')
            axs[1].set_title(f"target spectrogram (step {step})")
            self._add_mask_overlay(axs[1], torch.from_numpy(~mask_np))

            axs[2].imshow(lat_error, aspect='auto', origin='lower', cmap='hot')
            axs[2].set_title(f"latent error heatmap (masked) (step {step})")

            axs[3].imshow(context_repr_np, aspect='auto', origin='lower')
            axs[3].set_title(f"context embeddings (step {step})")

            axs[4].imshow(target_repr_np, aspect='auto', origin='lower')
            axs[4].set_title(f"target embeddings (step {step})")

            axs[5].imshow(pred_sequence_np, aspect='auto', origin='lower')
            axs[5].set_title(f"predicted embeddings (step {step})")

            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])

            plt.tight_layout()

        with Timer("latent_predictions_saving", debug=self.debug):
            fname = os.path.join(self.predictions_subfolder_path, f"latent_vis_{step}.png")
            plt.savefig(fname, dpi=100, bbox_inches='tight')
            plt.close(fig)

    def _add_mask_overlay(self, axis, mask_1d):
        if mask_1d.dim() > 1:
            mask_1d = mask_1d.any(dim=0)
        mask_1d = mask_1d.cpu().numpy()

        y_min, y_max = axis.get_ylim()
        bar_pos = y_max - 10
        for t, is_masked in enumerate(mask_1d):
            if is_masked:
                axis.add_patch(plt.Rectangle((t, bar_pos), 1, 10, edgecolor='none', facecolor='red'))

    def compute_embedding_stats(self, embeddings):
        with torch.no_grad():
            var_ = embeddings.var(dim=-1).mean().item()
            B,T,H = embeddings.shape
            flat = embeddings.reshape(-1,H)
            if flat.shape[0] > 1000:
                idx = torch.randperm(flat.shape[0])[:1000]
                flat = flat[idx]
            dmat = torch.cdist(flat, flat)
            avg_dist = dmat.mean().item()
            avg_norm = embeddings.norm(dim=-1).mean().item()
            return {
                'embedding_variance': var_,
                'avg_pairwise_distance': avg_dist,
                'avg_embedding_norm': avg_norm
            }

    def update_embedding_stats(self, c_repr, t_repr, p_repr):
        c = self.compute_embedding_stats(c_repr)
        t = self.compute_embedding_stats(t_repr)
        p = self.compute_embedding_stats(p_repr)

        self.embedding_stats_history['context_variance'].append(c['embedding_variance'])
        self.embedding_stats_history['context_distance'].append(c['avg_pairwise_distance'])
        self.embedding_stats_history['context_norm'].append(c['avg_embedding_norm'])

        self.embedding_stats_history['target_variance'].append(t['embedding_variance'])
        self.embedding_stats_history['target_distance'].append(t['avg_pairwise_distance'])
        self.embedding_stats_history['target_norm'].append(t['avg_embedding_norm'])

        self.embedding_stats_history['pred_variance'].append(p['embedding_variance'])
        self.embedding_stats_history['pred_distance'].append(p['avg_pairwise_distance'])
        self.embedding_stats_history['pred_norm'].append(p['avg_embedding_norm'])

    def get_network_stats(self, network, name):
        grad_norm=0
        weight_norm=0
        grad_max=0
        weight_max=0
        param_count=0

        for p in network.parameters():
            if p.grad is not None:
                gn = p.grad.norm().item()
                grad_norm += gn**2
                gmax = p.grad.abs().max().item()
                grad_max = max(grad_max, gmax)
            pn = p.norm().item()
            weight_norm += pn**2
            weight_max = max(weight_max, p.abs().max().item())
            param_count += p.numel()

        grad_norm = grad_norm**0.5
        weight_norm=weight_norm**0.5

        return {
            f"{name}_grad_norm": grad_norm,
            f"{name}_grad_max":  grad_max,
            f"{name}_weight_norm": weight_norm,
            f"{name}_weight_max": weight_max,
            f"{name}_param_count": param_count
        }

    def moving_average(self, values, window):
        if not len(values):
            return []
        alpha= 2/(window+1)
        out = [values[0]]
        for x in values[1:]:
            out.append(alpha*x + (1-alpha)*out[-1])
        return out

    def plot_results(self, save_plot=True, config=None):
        stats_file = os.path.join(self.experiment_dir, "training_statistics.json")
        with open(stats_file, 'r') as jf:
            training_stats = json.load(jf)

        # Training steps use full step count
        steps = list(range(len(training_stats['training_loss'])))
        
        # Embedding stats use validation intervals
        emb = training_stats['embedding_stats_history']
        num_emb_points = len(emb['context_variance'])
        steps_emb = [i*self.eval_interval for i in range(num_emb_points)]

        train_losses = training_stats['training_loss']
        val_losses   = training_stats['validation_loss']

        s_train = self.moving_average(train_losses, self.trailing_avg_window)
        s_val   = self.moving_average(val_losses,   self.trailing_avg_window)

        fig = plt.figure(figsize=(20,15))
        ax1 = fig.add_subplot(2,2,1)
        ax1.plot(steps, s_train, label='Smoothed Train Loss')
        ax1.plot(steps, s_val,   label='Smoothed Val Loss')
        ax1.legend()
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')

        ax2 = fig.add_subplot(2,2,2)
        ax2.plot(steps_emb, emb['context_variance'], label='Context Var')
        ax2.plot(steps_emb, emb['target_variance'],  label='Target Var')
        ax2.plot(steps_emb, emb['pred_variance'],    label='Pred Var')
        ax2.legend()
        ax2.set_title('Embedding Variance')

        ax3 = fig.add_subplot(2,2,3)
        ax3.plot(steps_emb, emb['context_distance'], label='Context Dist')
        ax3.plot(steps_emb, emb['target_distance'],  label='Target Dist')
        ax3.plot(steps_emb, emb['pred_distance'],    label='Pred Dist')
        ax3.legend()
        ax3.set_title('Avg Pairwise Distance')

        ax4 = fig.add_subplot(2,2,4)
        ax4.plot(steps_emb, emb['context_norm'], label='Context Norm')
        ax4.plot(steps_emb, emb['target_norm'],  label='Target Norm')
        ax4.plot(steps_emb, emb['pred_norm'],    label='Pred Norm')
        ax4.legend()
        ax4.set_title('Avg Embedding Norm')

        plt.tight_layout()
        if save_plot:
            plt.savefig(os.path.join(self.experiment_dir, 'training_plots.png'))
        else:
            plt.show()
        plt.close()

    def adjust_momentum(self, step):
        progress = step / self.max_steps
        new_m = self.model.ema_m + (1.0 - self.model.ema_m)*progress
        self.model.ema_m = new_m
        return new_m

    @timed_operation("visualize_stacked_layers")
    def visualize_stacked_layers(
        self, 
        step,
        context_spectrogram,
        target_spectrogram,
        context_intermediate,
        target_intermediate,
        mask
    ):
        # Move all tensors to CPU and convert to numpy at once
        with Timer("stacked_layers_data_preparation", debug=self.debug):
            context_spec_np = context_spectrogram[0].cpu().numpy()
            target_spec_np = target_spectrogram[0].cpu().numpy()
            mask_np = mask[0].cpu().numpy()
            context_inter_np = [layer[0].cpu().numpy().T for layer in context_intermediate]
            target_inter_np = [layer[0].cpu().numpy().T for layer in target_intermediate]

        with Timer("stacked_layers_figure_creation", debug=self.debug):
            num_layers = len(context_intermediate)
            fig, axs = plt.subplots(num_layers + 1, 2, figsize=(16, 3 * (num_layers + 1)))

        with Timer("stacked_layers_plotting", debug=self.debug):
            # Plot input spectrograms
            axs[0, 0].imshow(context_spec_np, aspect='auto', origin='lower')
            axs[0, 0].set_title(f"Context Spectrogram (Input) - Step {step}")
            self._add_mask_overlay(axs[0, 0], torch.from_numpy(mask_np))
            
            axs[0, 1].imshow(target_spec_np, aspect='auto', origin='lower')
            axs[0, 1].set_title(f"Target Spectrogram (Input) - Step {step}")
            self._add_mask_overlay(axs[0, 1], torch.from_numpy(~mask_np))
            
            # Determine layer types
            attention_blocks = self.model.context_encoder.attention_blocks
            layer_names = ["Initial Encoding"]
            
            for i, block in enumerate(attention_blocks):
                block_name = block.__class__.__name__
                if "LocalAttention" in block_name:
                    window_size = block.window_size
                    layer_names.append(f"Local Attention (window={window_size})")
                elif "GlobalAttention" in block_name:
                    stride = block.global_stride
                    layer_names.append(f"Global Attention (stride={stride})")
                else:
                    layer_names.append(f"Layer {i+1}")
            
            while len(layer_names) < num_layers + 1:
                layer_names.append(f"Layer {len(layer_names)}")
            
            # Plot all layers at once
            for i in range(num_layers):
                axs[i+1, 0].imshow(context_inter_np[i], aspect='auto', origin='lower')
                axs[i+1, 0].set_title(f"Context: {layer_names[i+1]} - Step {step}")
                
                axs[i+1, 1].imshow(target_inter_np[i], aspect='auto', origin='lower')
                axs[i+1, 1].set_title(f"Target: {layer_names[i+1]} - Step {step}")
            
            # Clean up axes
            for row in axs:
                for ax in row:
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            plt.tight_layout()

        with Timer("stacked_layers_saving", debug=self.debug):
            fname = os.path.join(self.predictions_subfolder_path, f"all_layers_stacked_{step}.png")
            plt.savefig(fname, dpi=100, bbox_inches='tight')
            plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--test_dir',  type=str, required=True)
    parser.add_argument('--steps',     type=int, default=50000)
    parser.add_argument('--batch_size',type=int, default=64)
    parser.add_argument('--lr',        type=float, default=1e-4)
    parser.add_argument('--name',      type=str, required=True)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--input_dim', type=int, default=513)
    parser.add_argument('--hidden_dim',type=int, default=256)
    parser.add_argument('--num_layers',type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout',   type=float, default=0.0)
    parser.add_argument('--mlp_dim',   type=int, default=512)
    parser.add_argument('--pred_hidden_dim', type=int, default=256)
    parser.add_argument('--pred_num_layers', type=int, default=2)
    parser.add_argument('--pred_num_heads',  type=int, default=4)
    parser.add_argument('--pred_mlp_dim',    type=int, default=512)
    parser.add_argument('--max_seq_len',     type=int, default=500)
    parser.add_argument('--overfit_batch',   action='store_true')
    parser.add_argument('--mask_ratio',      type=float, default=0.3)
    parser.add_argument('--verbose',         action='store_true')
    parser.add_argument('--encoder_lr',      type=float, default=1e-4)
    parser.add_argument('--predictor_lr',    type=float, default=1e-4)
    parser.add_argument('--decoder_lr',      type=float, default=1e-4)
    parser.add_argument('--freeze_encoder_steps', type=int, default=0)
    parser.add_argument('--freeze_decoder_steps', type=int, default=0)
    parser.add_argument('--patience',        type=int, default=4)
    parser.add_argument('--ema_momentum',    type=float, default=0.9)
    parser.add_argument('--continue_training', action='store_true', help='Continue training from checkpoint')
    parser.add_argument('--zero_predictor_input', action='store_true',
                      help='If set, the predictor input is zeroed out at masked positions')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug mode for additional logging and model inspection')
    
    # Attention configuration - use flexible architecture parameter
    parser.add_argument('--architecture', type=str, default="local:8,global:100",
                       help='Comma-separated architecture specification: "type:param,type:param,..."')
    
    args = parser.parse_args()

    root_exp_dir = "experiments"
    archive_dir  = os.path.join(root_exp_dir, "archive")
    experiment_dir = os.path.join(root_exp_dir, args.name)
    os.makedirs(archive_dir, exist_ok=True)

    # Initialize timing_stats before anything else
    timing_stats.set_experiment_name(args.name)
    timing_stats.log_dir = root_exp_dir

    if os.path.exists(experiment_dir) and not args.continue_training:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_name = f"{args.name}_{timestamp}"
        archived_path = os.path.join(archive_dir, archived_name)
        shutil.move(experiment_dir, archived_path)
        print(f"Archived existing experiment to: {archived_path}")

    os.makedirs(experiment_dir, exist_ok=True)
    weights_dir = os.path.join(experiment_dir, 'saved_weights')
    os.makedirs(experiment_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dl_train = DataLoader(
        BirdJEPA_Dataset(data_dir=args.train_dir, segment_len=args.max_seq_len, verbose=False),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(
            batch,
            segment_length=args.max_seq_len,
            mask_p=args.mask_ratio,
            verbose=False
        )
    )
    dl_test = DataLoader(
        BirdJEPA_Dataset(data_dir=args.test_dir, segment_len=args.max_seq_len, verbose=False),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(
            batch,
            segment_length=args.max_seq_len,
            mask_p=args.mask_ratio,
            verbose=False
        )
    )

    from model import BirdJEPA
    
    # Parse the architecture string to create blocks_config
    blocks_config = []
    for block_spec in args.architecture.split(','):
        parts = block_spec.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid block specification: {block_spec}")
        
        block_type, param = parts
        if block_type.lower() == "local":
            blocks_config.append({"type": "local", "window_size": int(param)})
        elif block_type.lower() == "global":
            blocks_config.append({"type": "global", "stride": int(param)})
        else:
            raise ValueError(f"Unknown block type: {block_type}")

    # Create the model with only the necessary parameters
    model = BirdJEPA(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        mlp_dim=args.mlp_dim,
        pred_hidden_dim=args.pred_hidden_dim,
        pred_num_layers=args.pred_num_layers,
        pred_num_heads=args.pred_num_heads,
        pred_mlp_dim=args.pred_mlp_dim,
        max_seq_len=args.max_seq_len,
        zero_predictor_input=args.zero_predictor_input,
        debug=args.debug,
        blocks_config=blocks_config
    ).to(device)

    config = {
        "input_dim":      args.input_dim,
        "max_seq_len":    args.max_seq_len,
        "mask_ratio":     args.mask_ratio,
        "hidden_dim":     args.hidden_dim,
        "num_layers":     args.num_layers,
        "num_heads":      args.num_heads,
        "dropout":        args.dropout,
        "mlp_dim":        args.mlp_dim,
        "pred_hidden_dim": args.pred_hidden_dim,
        "pred_num_layers": args.pred_num_layers,
        "pred_num_heads":  args.pred_num_heads,
        "pred_mlp_dim":    args.pred_mlp_dim,
        "batch_size":      args.batch_size,
        "learning_rate":   args.lr,
        "max_steps":       args.steps,
        "eval_interval":   args.eval_interval,
        "experiment_name": args.name,
        "train_dir":       args.train_dir,
        "test_dir":        args.test_dir,
        "device": str(device),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "debug": args.debug,
        "architecture": args.architecture
    }

    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    try:
        if args.continue_training:
            model, checkpoint, config = load_model(os.path.join("experiments", args.name), return_checkpoint=True)
            model = model.to(device)
            
            trainer = ModelTrainer(
                model=model,
                train_loader=dl_train,
                test_loader=dl_test,
                device=device,
                max_steps=args.steps,
                eval_interval=args.eval_interval,
                save_interval=500,
                weights_save_dir=weights_dir,
                experiment_dir=experiment_dir,
                verbose=args.verbose,
                overfit_single_batch=args.overfit_batch,
                encoder_lr=config.get('encoder_lr', args.encoder_lr),
                predictor_lr=config.get('predictor_lr', args.predictor_lr),
                decoder_lr=config.get('decoder_lr', args.decoder_lr),
                freeze_encoder_steps=args.freeze_encoder_steps,
                freeze_decoder_steps=args.freeze_decoder_steps,
                debug=args.debug,
                lr=args.lr
            )
            
            last_step, training_stats, raw_loss_list, raw_val_loss_list = trainer.load_model(checkpoint=checkpoint)
            trainer.train(
                continue_training=True,
                training_stats=training_stats,
                last_step=last_step,
                raw_loss_list=raw_loss_list,
                raw_val_loss_list=raw_val_loss_list
            )
        else:
            trainer = ModelTrainer(
                model=model,
                train_loader=dl_train,
                test_loader=dl_test,
                device=device,
                max_steps=args.steps,
                eval_interval=args.eval_interval,
                save_interval=500,
                weights_save_dir=weights_dir,
                experiment_dir=experiment_dir,
                verbose=args.verbose,
                overfit_single_batch=args.overfit_batch,
                encoder_lr=args.encoder_lr,
                predictor_lr=args.predictor_lr,
                decoder_lr=args.decoder_lr,
                freeze_encoder_steps=args.freeze_encoder_steps,
                freeze_decoder_steps=args.freeze_decoder_steps,
                debug=args.debug,
                lr=args.lr
            )
            
            trainer.train()
            trainer.plot_results(save_plot=True)
    finally:
        # Restore stdout and close log file
        if args.debug and 'trainer' in locals() and hasattr(trainer, 'log_capture'):
            sys.stdout = trainer.log_capture.terminal
            trainer.log_capture.close()
