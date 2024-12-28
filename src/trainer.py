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
        lr=None   # optional single-lr param from older code
    ):
        self.model = model
        self.model.ema_m = ema_momentum
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

    def train(self, continue_training=False, training_stats=None, last_step=0, raw_loss_list=None, raw_val_loss_list=None):
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

        while step < self.max_steps:
            try:
                if not self.overfit_single_batch:
                    batch = next(train_iter)
                    val_batch = next(test_iter)
            except StopIteration:
                if not self.overfit_single_batch:
                    train_iter = iter(self.train_iter)
                    test_iter  = iter(self.test_iter)
                    continue

            self.adjust_momentum(step)
            
            full_spect, targ_spect, cont_spect, labels, mask, file_names = batch
            cont_spect = cont_spect.to(self.device)
            targ_spect = targ_spect.to(self.device)
            mask       = mask.to(self.device)

            self.model.train()

            self.encoder_optimizer.zero_grad(set_to_none=True)
            self.predictor_optimizer.zero_grad(set_to_none=True)
            self.decoder_optimizer.zero_grad(set_to_none=True)

            with autocast():
                loss, diff, pred, target_repr, context_repr = self.model.compute_latent_loss(
                    cont_spect, targ_spect, mask
                )

            scaler.scale(loss).backward()

            if step >= self.freeze_encoder_steps and has_gradients(self.encoder_optimizer):
                scaler.unscale_(self.encoder_optimizer)
                scaler.step(self.encoder_optimizer)

            if step >= self.freeze_decoder_steps and has_gradients(self.decoder_optimizer):
                scaler.unscale_(self.decoder_optimizer)
                scaler.step(self.decoder_optimizer)

            if has_gradients(self.predictor_optimizer):
                scaler.unscale_(self.predictor_optimizer)
                scaler.step(self.predictor_optimizer)

            scaler.update()

            try:
                val_full, val_target, val_cont, val_labels, val_mask, val_fnames = val_batch
            except StopIteration:
                test_iter = iter(self.test_iter)
                val_batch = next(test_iter)
                val_full, val_target, val_cont, val_labels, val_mask, val_fnames = val_batch

            val_loss, _ = self.validate_model(
                step,
                context_spec=val_cont.to(self.device),
                target_spec=val_target.to(self.device),
                mask=val_mask.to(self.device),
                vocalization=val_labels.to(self.device)
            )

            raw_loss_list.append(loss.item())
            raw_val_loss_list.append(val_loss)

            # record them in stats for continuity
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

                print(f"\nStep [{step}/{self.max_steps}] - "
                      f"Smoothed Train Loss: {smoothed_training_loss:.4f}, "
                      f"Smoothed Val Loss: {smoothed_val_loss:.4f}, "
                      f"Vars [C/T]: {c_var:.4f}/{t_var:.4f} - "
                      f"GradNorm: {gnorm:.2e} - "
                      f"LRs: enc={enc_lr:.2e}, pred={pred_lr:.2e}, dec={dec_lr:.2e}")

                if smoothed_val_loss < best_val_loss:
                    best_val_loss = smoothed_val_loss
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1

                if self.early_stopping and steps_since_improvement >= self.patience:
                    print(f"\nEarly stopping triggered at step {step}. no improvement for {self.patience} intervals.")
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
            self.model.update_ema()

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
        return raw_loss_list, raw_val_loss_list

    def validate_model(self, step, context_spec, target_spec, mask, vocalization):
        self.model.eval()
        with torch.no_grad():
            loss, diff, pred_sequence, target_repr, context_repr = self.model.compute_latent_loss(
                context_spec, target_spec, mask
            )
            c_stats = self.compute_embedding_stats(context_repr)
            t_stats = self.compute_embedding_stats(target_repr)
            p_stats = self.compute_embedding_stats(pred_sequence)
            self.update_embedding_stats(context_repr, target_repr, pred_sequence)

            if step % self.eval_interval == 0:
                print("\nEmbedding Statistics:")
                print(f"Context var={c_stats['embedding_variance']:.4f}, dist={c_stats['avg_pairwise_distance']:.4f}, norm={c_stats['avg_embedding_norm']:.4f}")
                print(f"Target  var={t_stats['embedding_variance']:.4f}, dist={t_stats['avg_pairwise_distance']:.4f}, norm={t_stats['avg_embedding_norm']:.4f}")
                print(f"Pred    var={p_stats['embedding_variance']:.4f}, dist={p_stats['avg_pairwise_distance']:.4f}, norm={p_stats['avg_embedding_norm']:.4f}")

                self.visualize_latent_predictions(
                    step=step,
                    context_spectrogram=context_spec,
                    target_spectrogram=target_spec,
                    context_repr=context_repr,
                    pred_sequence=pred_sequence,
                    target_repr=target_repr,
                    mask=mask
                )

            return loss.item(), {}

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
        fig, axs = plt.subplots(2, 3, figsize=(20,10))
        axs = axs.flatten()

        img0 = axs[0].imshow(context_spectrogram[0].cpu().numpy(), aspect='auto', origin='lower')
        axs[0].set_title("context spectrogram")
        self._add_mask_overlay(axs[0], mask[0])

        img1 = axs[1].imshow(target_spectrogram[0].cpu().numpy(), aspect='auto', origin='lower')
        axs[1].set_title("target spectrogram")
        self._add_mask_overlay(axs[1], ~mask[0])

        mask_expanded = mask.unsqueeze(-1)
        lat_error = ((pred_sequence - target_repr)**2 * mask_expanded)[0].cpu().numpy().T
        axs[2].imshow(lat_error, aspect='auto', origin='lower', cmap='hot')
        axs[2].set_title("latent error heatmap (masked)")

        c_repr_disp = context_repr[0].cpu().numpy().T
        axs[3].imshow(c_repr_disp, aspect='auto', origin='lower')
        axs[3].set_title("context embeddings")

        t_repr_disp = target_repr[0].cpu().numpy().T
        axs[4].imshow(t_repr_disp, aspect='auto', origin='lower')
        axs[4].set_title("target embeddings")

        p_repr_disp = pred_sequence[0].cpu().numpy().T
        axs[5].imshow(p_repr_disp, aspect='auto', origin='lower')
        axs[5].set_title("predicted embeddings")

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        fname = os.path.join(self.predictions_subfolder_path, f"latent_vis_{step}.png")
        plt.savefig(fname, dpi=100)
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

        steps = list(range(len(training_stats['training_loss'])))
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

        emb = training_stats['embedding_stats_history']
        ax2 = fig.add_subplot(2,2,2)
        ax2.plot(steps, emb['context_variance'], label='Context Var')
        ax2.plot(steps, emb['target_variance'],  label='Target Var')
        ax2.plot(steps, emb['pred_variance'],    label='Pred Var')
        ax2.legend()
        ax2.set_title('Embedding Variance')

        ax3 = fig.add_subplot(2,2,3)
        ax3.plot(steps, emb['context_distance'], label='Context Dist')
        ax3.plot(steps, emb['target_distance'],  label='Target Dist')
        ax3.plot(steps, emb['pred_distance'],    label='Pred Dist')
        ax3.legend()
        ax3.set_title('Avg Pairwise Distance')

        ax4 = fig.add_subplot(2,2,4)
        ax4.plot(steps, emb['context_norm'], label='Context Norm')
        ax4.plot(steps, emb['target_norm'],  label='Target Norm')
        ax4.plot(steps, emb['pred_norm'],    label='Pred Norm')
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
    
    args = parser.parse_args()

    root_exp_dir = "experiments"
    archive_dir  = os.path.join(root_exp_dir, "archive")
    experiment_dir = os.path.join(root_exp_dir, args.name)
    os.makedirs(archive_dir, exist_ok=True)

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
        zero_predictor_input=args.zero_predictor_input
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
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

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
            lr=args.lr
        )

        trainer.train()
        trainer.plot_results(save_plot=True)
