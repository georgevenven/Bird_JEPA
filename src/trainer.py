# trainer.py
import os
import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import json
from torch.cuda.amp import autocast, GradScaler
from model import BirdJEPA
from data_class import BirdJEPA_Dataset, collate_fn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import shutil
from datetime import datetime

class ModelTrainer:
    def __init__(self, model, train_loader, test_loader, optimizer, device, 
                 max_steps=1000, eval_interval=100, save_interval=500,
                 weights_save_dir="saved_weights", experiment_dir="experiments",
                 trailing_avg_window=100, early_stopping=True, patience=10,
                 verbose=False, overfit_single_batch=False):
        self.model = model
        self.train_iter = train_loader
        self.test_iter = test_loader
        self.optimizer = optimizer
        self.device = device
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.trailing_avg_window = trailing_avg_window
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose
        self.overfit_single_batch = overfit_single_batch
        
        # Print model parameter counts
        context_params = sum(p.numel() for p in self.model.context_encoder.parameters())
        target_params = sum(p.numel() for p in self.model.target_encoder.parameters())
        predictor_params = sum(p.numel() for p in self.model.predictor.parameters())
        decoder_params = sum(p.numel() for p in self.model.decoder.parameters())
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print("\nModel Parameter Counts:")
        print(f"Context Encoder:  {context_params:,} parameters")
        print(f"Target Encoder:   {target_params:,} parameters")
        print(f"Predictor:        {predictor_params:,} parameters")
        print(f"Decoder:          {decoder_params:,} parameters")
        print(f"Total:            {total_params:,} parameters\n")
        
        # Setup directories
        self.experiment_dir = experiment_dir
        self.weights_save_dir = weights_save_dir
        os.makedirs(self.weights_save_dir, exist_ok=True)
        
        # Create predictions subfolder
        self.predictions_subfolder_path = os.path.join(self.experiment_dir, 'predictions')
        os.makedirs(self.predictions_subfolder_path, exist_ok=True)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_steps)

    def embedding_variance(self, embeddings):
        # Calculate variance of embeddings
        # embeddings: (B,T,H)
        return embeddings.var().item()

    def save_model(self, step, training_stats):
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        checkpoint_path = os.path.join(self.weights_save_dir, f'checkpoint_{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Convert tensors to Python native types for JSON serialization
        json_safe_stats = {}
        for key, value in training_stats.items():
            if isinstance(value, (list, tuple)) and len(value) > 0 and torch.is_tensor(value[0]):
                json_safe_stats[key] = [float(v.cpu().detach()) if torch.is_tensor(v) else float(v) for v in value]
            elif torch.is_tensor(value):
                json_safe_stats[key] = float(value.cpu().detach())
            else:
                json_safe_stats[key] = value
        
        # Save training statistics
        stats_filename = "training_statistics.json"
        stats_filepath = os.path.join(self.experiment_dir, stats_filename)
        with open(stats_filepath, 'w') as json_file:
            json.dump(json_safe_stats, json_file)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['step'], checkpoint['training_stats']

    def create_large_canvas(self, context_outputs, target_outputs=None, image_idx=0, spec_shape=None, 
                           context_spec=None, target_spec=None):
        context_layers = context_outputs["layer_outputs"].shape[0]
        total_rows = context_layers + 1  # +1 for input spectrograms
        
        # Use 2:1 aspect ratio for each subplot
        fig, axes = plt.subplots(total_rows, 2, figsize=(20, 5*total_rows))
        
        # Plot input spectrograms
        axes[0,0].imshow(context_spec.cpu()[image_idx].numpy(), aspect='auto', origin='lower')
        axes[0,0].set_title("Input to Context Encoder")
        axes[0,1].imshow(target_spec.cpu()[image_idx].numpy(), aspect='auto', origin='lower')
        axes[0,1].set_title("Input to Target Encoder")
        
        # Plot intermediate layers - transpose to (H,T)
        for i in range(context_layers):
            # Context encoder layers
            c_layer = context_outputs["layer_outputs"][i][image_idx].cpu().numpy().T
            axes[i+1,0].imshow(c_layer, aspect='auto', origin='lower')
            axes[i+1,0].set_title(f"Context Encoder Layer {i}")
            
            # Target encoder layers (now directly using tensor)
            if target_outputs is not None:
                t_layer = target_outputs[i][image_idx].cpu().numpy().T
                axes[i+1,1].imshow(t_layer, aspect='auto', origin='lower')
                axes[i+1,1].set_title(f"Target Encoder Layer {i}")
            else:
                axes[i+1,1].axis('off')
        
        # Remove legends and adjust layout
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()

    def visualize_mse(self, output, mask, spec, step, full_spectrogram=None):
        # Use 2:1 aspect ratio for each subplot
        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        axs = axs.flatten()

        # Plot 1: Original spectrogram with mask overlay
        im = axs[0].imshow(spec[0].cpu().numpy(), aspect='auto', origin='lower')
        axs[0].set_title('Masked Input Spectrogram', fontsize=35, pad=20)
        self._add_mask_overlay(axs[0], mask[0])

        # Plot 2: Model's prediction
        im = axs[1].imshow(output[0].cpu().detach().numpy(), aspect='auto', origin='lower')
        axs[1].set_title('Model Prediction', fontsize=35, pad=20)
        self._add_mask_overlay(axs[1], mask[0])

        # Plot 3: Empty for now
        axs[2].set_visible(False)

        # Plot 4: Error heatmap
        error_map = (output[0] - spec[0]).cpu().detach().numpy() ** 2
        im = axs[3].imshow(error_map, aspect='auto', origin='lower', cmap='hot')
        axs[3].set_title('Prediction Error Heatmap', fontsize=35, pad=20)

        # Remove legends and ticks
        for ax in axs:
            if ax.get_visible():
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(self.predictions_subfolder_path, f'MSE_Visualization_{step}.png'), format="png")
        plt.close(fig)

    def _add_mask_overlay(self, axis, mask):
        """
        Add mask overlay to plot
        Args:
            axis: matplotlib axis
            mask: tensor of shape (T) or (D, T)
        """
        # If mask is 2D (D,T), reduce to 1D (T)
        if mask.dim() > 1:
            masked_tokens = mask.any(dim=0)  # Reduce along frequency dimension if present
        else:
            masked_tokens = mask
        
        y_min, y_max = axis.get_ylim()
        mask_bar_position = y_max - 15
        
        # Convert to numpy for matplotlib
        masked_tokens = masked_tokens.cpu().numpy()
        
        # Iterate over time steps
        for t in range(len(masked_tokens)):
            if masked_tokens[t]:
                axis.add_patch(plt.Rectangle((t, mask_bar_position), 1, 15, 
                                           edgecolor='none', facecolor='red'))

    def visualize_masked_predictions(self, step, context_spec, target_spec, output, mask, all_outputs):
        self.model.eval()
        with torch.no_grad():
            # Visualize MSE and predictions
            self.visualize_mse(output=output, mask=mask, spec=target_spec, step=step)
            
            # Create visualization with both context and target - pass target_outputs directly
            self.create_large_canvas(
                context_outputs=all_outputs,
                target_outputs=all_outputs["target_outputs"], 
                image_idx=0, 
                spec_shape=context_spec.shape,
                context_spec=context_spec, 
                target_spec=target_spec
            )
            
            plt.savefig(os.path.join(self.predictions_subfolder_path, 
                       f'Intermediate_Outputs_{step}.png'), format="png")
            plt.close()

    def visualize_latent_predictions(self, step, context_repr, target_repr, pred_sequence, mask):
        """Visualize the latent space predictions"""
        # Use 2:1 aspect ratio for each subplot
        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        axs = axs.flatten()

        # Plot 1: Target embeddings
        im = axs[0].imshow(target_repr[0].cpu().numpy().T, aspect='auto', origin='lower')
        axs[0].set_title('Target Embeddings', fontsize=35, pad=20)
        self._add_mask_overlay(axs[0], mask[0])

        # Plot 2: Predicted embeddings
        im = axs[1].imshow(pred_sequence[0].cpu().detach().numpy().T, aspect='auto', origin='lower')
        axs[1].set_title('Predicted Embeddings', fontsize=35, pad=20)
        self._add_mask_overlay(axs[1], mask[0])

        # Plot 3: Error heatmap (only for masked regions)
        mask_expanded = mask[0].unsqueeze(-1).expand(-1, pred_sequence.shape[-1])
        error_map = ((pred_sequence[0] - target_repr[0]) * mask_expanded).cpu().detach().numpy().T ** 2
        im = axs[2].imshow(error_map, aspect='auto', origin='lower', cmap='hot')
        axs[2].set_title('Prediction Error Heatmap (Masked Regions)', fontsize=35, pad=20)

        # Plot 4: Context embeddings
        im = axs[3].imshow(context_repr[0].cpu().numpy().T, aspect='auto', origin='lower')
        axs[3].set_title('Context Embeddings', fontsize=35, pad=20)

        # Remove legends and ticks
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(self.predictions_subfolder_path, f'Latent_Predictions_{step}.png'), format="png")
        plt.close(fig)

    def visualize_decoder_reconstruction(self, step, original_spec, decoded_pred, mask):
        """Visualize the decoder's reconstruction"""
        # Ensure decoded_pred is in the right shape (B, D, T)
        if decoded_pred.shape[1] != original_spec.shape[1]:
            decoded_pred = decoded_pred.transpose(1, 2)
        
        # Use 2:1 aspect ratio for each subplot
        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        axs = axs.flatten()

        # Plot 1: Original spectrogram
        im = axs[0].imshow(original_spec[0].cpu().numpy(), aspect='auto', origin='lower')
        axs[0].set_title('Original Spectrogram', fontsize=35, pad=20)
        self._add_mask_overlay(axs[0], mask[0])

        # Plot 2: Decoded prediction
        im = axs[1].imshow(decoded_pred[0].cpu().detach().numpy(), aspect='auto', origin='lower')
        axs[1].set_title('Decoded Prediction', fontsize=35, pad=20)
        self._add_mask_overlay(axs[1], mask[0])

        # Plot 3: Error heatmap for full reconstruction
        error_map = (decoded_pred[0] - original_spec[0]).cpu().detach().numpy() ** 2
        im = axs[2].imshow(error_map, aspect='auto', origin='lower', cmap='hot')
        axs[2].set_title('Full Reconstruction Error', fontsize=35, pad=20)

        # Plot 4: Error heatmap for masked regions only
        masked_error = error_map * mask[0].cpu().numpy()
        im = axs[3].imshow(masked_error, aspect='auto', origin='lower', cmap='hot')
        axs[3].set_title('Masked Regions Error', fontsize=35, pad=20)

        # Remove legends and ticks
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(self.predictions_subfolder_path, f'Decoder_Reconstruction_{step}.png'), format="png")
        plt.close(fig)

    def validate_model(self, step, context_spec, target_spec, vocalization):
        self.model.eval()
        with torch.no_grad():
            # Clean context input by zeroing masked positions
            context_clean = torch.where(context_spec == -1.0, 
                                      torch.zeros_like(context_spec), 
                                      context_spec)
            
            context_repr, context_intermediate = self.model.context_encoder(context_clean)
            target_repr, target_intermediate = self.model.target_encoder(target_spec)
            
            # Extract mask from context_spec (where values are -1.0)
            mask = (context_spec == -1.0).any(dim=1)  # (B,T)
            
            # Pass mask to predictor
            pred_sequence = self.model.predictor(context_repr, mask=mask)
            
            # Calculate latent space loss properly across all batches
            masked_latent_loss = ((pred_sequence - target_repr)**2 * mask.unsqueeze(-1)).mean()
            
            # Get decoder predictions
            decoded_pred = self.model.decoder(pred_sequence)
            decoded_pred = decoded_pred.transpose(1, 2)
            
            # Calculate embedding variances
            context_var = self.embedding_variance(context_repr)
            target_var = self.embedding_variance(target_repr)
            
            # Generate visualizations at eval_interval
            if step % self.eval_interval == 0:
                # Latent predictions visualization
                self.visualize_latent_predictions(
                    step, context_repr=context_repr, target_repr=target_repr,
                    pred_sequence=pred_sequence, mask=mask
                )
                
                # Encoder states visualization
                context_outputs_dict = {"layer_outputs": torch.stack(context_intermediate, dim=0),
                                      "target_outputs": torch.stack(target_intermediate, dim=0)}
                
                self.create_large_canvas(
                    context_outputs=context_outputs_dict,
                    target_outputs=target_intermediate,
                    image_idx=0,
                    spec_shape=context_spec.shape,
                    context_spec=context_spec,
                    target_spec=target_spec
                )
                plt.savefig(os.path.join(self.predictions_subfolder_path, 
                           f'Encoders_Activations_{step}.png'), format="png")
                plt.close()
                
                # Decoder reconstruction visualization
                if step % 1000 == 0:
                    self.visualize_decoder_reconstruction(
                        step, original_spec=target_spec,
                        decoded_pred=decoded_pred, mask=mask
                    )
            
            # Calculate sequence accuracies (for tracking)
            mask_expanded = mask.unsqueeze(1)
            masked_seq_acc = ((decoded_pred - target_spec)**2 * mask_expanded.float()).mean()
            unmasked_seq_acc = ((decoded_pred - target_spec)**2 * (~mask_expanded).float()).mean()
            
            return masked_latent_loss.item(), masked_seq_acc.item(), unmasked_seq_acc.item()

    def train_decoder(self):
        """Periodically train decoder on unmasked data"""
        self.model.context_encoder.eval()  # Freeze context encoder
        self.model.decoder.train()  # Set decoder to training mode
        decoder_optimizer = torch.optim.Adam(self.model.decoder.parameters(), lr=1e-4)
        
        for batch in self.train_iter:  # Use a subset of data if needed
            full_spec = batch[0].to(self.device)  # shape: (B, D, T)
            
            with torch.no_grad():
                # Get embeddings from frozen encoder
                embeddings, _ = self.model.context_encoder(full_spec)  # shape: (B, T, H)
            
            # Forward pass through decoder (with gradients)
            decoder_optimizer.zero_grad()
            decoded = self.model.decoder(embeddings)  # shape: (B, T, D)
            decoded = decoded.transpose(1, 2)  # Now shape: (B, D, T)
            
            # Compute loss
            decoder_loss = F.mse_loss(decoded, full_spec)
            
            # Backward pass
            decoder_loss.backward()
            decoder_optimizer.step()
            
            # Only train on one batch
            break

    def moving_average(self, values, window):
        if len(values) < window:
            return []
        
        # Convert tensors to CPU numpy arrays if needed
        numpy_values = []
        for v in values:
            if torch.is_tensor(v):
                numpy_values.append(v.detach().cpu().numpy())
            else:
                numpy_values.append(v)
        
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(numpy_values, weights, 'valid')
        return sma.tolist()

    def get_network_stats(self, network, name):
        """Get gradient and weight statistics for a network"""
        grad_norm = 0
        weight_norm = 0
        grad_max = 0
        weight_max = 0
        param_count = 0
        
        for p in network.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
                grad_max = max(grad_max, p.grad.abs().max().item())
            weight_norm += p.norm().item() ** 2
            weight_max = max(weight_max, p.abs().max().item())
            param_count += p.numel()
        
        grad_norm = grad_norm ** 0.5
        weight_norm = weight_norm ** 0.5
        
        return {
            f"{name}_grad_norm": grad_norm,
            f"{name}_grad_max": grad_max,
            f"{name}_weight_norm": weight_norm,
            f"{name}_weight_max": weight_max,
            f"{name}_param_count": param_count
        }

    def train(self, continue_training=False, training_stats=None, last_step=0):
        step = last_step + 1 if continue_training else 0
        scaler = GradScaler()

        if continue_training:
            raw_loss_list = training_stats.get('training_loss', [])
            raw_val_loss_list = training_stats.get('validation_loss', [])
            raw_masked_seq_acc_list = training_stats.get('masked_seq_acc', [])
            raw_unmasked_seq_acc_list = training_stats.get('unmasked_seq_acc', [])
            steps_since_improvement = training_stats.get('steps_since_improvement', 0)
            best_val_loss = training_stats.get('best_val_loss', float('inf'))
        else:
            raw_loss_list = []
            raw_val_loss_list = []
            raw_masked_seq_acc_list = []
            raw_unmasked_seq_acc_list = []
            steps_since_improvement = 0
            best_val_loss = float('inf')

        train_iter = iter(self.train_iter)
        test_iter = iter(self.test_iter)

        # Get single batch if overfitting
        if self.overfit_single_batch:
            train_batch = next(iter(self.train_iter))
            val_batch = train_batch  # Use same batch for validation
            print("\nOverfitting on single batch...")
            
        while step < self.max_steps:
            try:
                if not self.overfit_single_batch:
                    train_batch = next(train_iter)
                    val_batch = next(test_iter)
            except StopIteration:
                if not self.overfit_single_batch:
                    train_iter = iter(self.train_iter)
                    test_iter = iter(self.test_iter)
                    continue

            # Unpack batches
            full_spectrogram, target_spectrogram, context_spectrogram, ground_truth_labels, vocalization, file_names = train_batch
            val_full_spect, val_target_spect, val_context_spect, val_labels, val_vocalization, val_file_names = val_batch
            
            # Move to device
            context_spectrogram = context_spectrogram.to(self.device)
            target_spectrogram = target_spectrogram.to(self.device)
            val_context_spect = val_context_spect.to(self.device)
            val_target_spect = val_target_spect.to(self.device)
            val_vocalization = val_vocalization.to(self.device) if torch.is_tensor(val_vocalization) else val_vocalization

            # Training step
            self.model.train()
            with autocast():
                loss = self.model.training_step(context_spectrogram, target_spectrogram)

            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.scheduler.step()

            # Periodically train decoder
            if step % 1000 == 0:
                self.train_decoder()

            # Validation step
            val_loss, masked_seq_acc, unmasked_seq_acc = self.validate_model(
                step, context_spec=val_context_spect, 
                target_spec=val_target_spect, 
                vocalization=val_vocalization
            )

            # Update statistics
            raw_loss_list.append(loss.item())
            raw_val_loss_list.append(val_loss)
            raw_masked_seq_acc_list.append(masked_seq_acc)
            raw_unmasked_seq_acc_list.append(unmasked_seq_acc)

            if step % self.eval_interval == 0:
                # Calculate variances
                with torch.no_grad():
                    context_repr, _ = self.model.context_encoder(context_spectrogram)
                    target_repr, _ = self.model.target_encoder(target_spectrogram)
                    context_var = self.embedding_variance(context_repr)
                    target_var = self.embedding_variance(target_repr)

                # Calculate smoothed losses
                if len(raw_loss_list) >= self.trailing_avg_window:
                    smoothed_training_loss = self.moving_average(raw_loss_list, self.trailing_avg_window)[-1]
                    smoothed_val_loss = self.moving_average(raw_val_loss_list, self.trailing_avg_window)[-1]
                else:
                    smoothed_training_loss = loss.item()
                    smoothed_val_loss = val_loss

                # Single consolidated print statement
                print(f"\nStep [{step}/{self.max_steps}] - Train Loss: {smoothed_training_loss:.4f}, "
                      f"Val Loss: {smoothed_val_loss:.4f}, Vars [C/T]: {context_var:.4f}/{target_var:.4f}")

                if smoothed_val_loss < best_val_loss:
                    best_val_loss = smoothed_val_loss
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1

                if self.early_stopping and steps_since_improvement >= self.patience:
                    print(f"\nEarly stopping triggered at step {step}. No improvement for {self.patience} intervals.")
                    break

            if step % self.save_interval == 0:
                training_stats = {
                    'training_loss': raw_loss_list,
                    'validation_loss': raw_val_loss_list,
                    'masked_seq_acc': raw_masked_seq_acc_list,
                    'unmasked_seq_acc': raw_unmasked_seq_acc_list,
                    'best_val_loss': best_val_loss,
                    'steps_since_improvement': steps_since_improvement
                }
                self.save_model(step, training_stats)

            step += 1
            # Update target encoder EMA
            self.model.update_ema()

        return raw_loss_list, raw_val_loss_list

    def plot_results(self, save_plot=True, config=None):
        stats_filename = "training_statistics.json"
        stats_filepath = os.path.join(self.experiment_dir, stats_filename)
        
        with open(stats_filepath, 'r') as json_file:
            training_stats = json.load(json_file)
        
        steps = list(range(len(training_stats['training_loss'])))
        training_losses = training_stats['training_loss']
        validation_losses = training_stats['validation_loss']

        masked_seq_acc = training_stats['masked_seq_acc']
        unmasked_seq_acc = training_stats['unmasked_seq_acc']

        smoothed_training_losses = self.moving_average(training_losses, self.trailing_avg_window)
        smoothed_validation_losses = self.moving_average(validation_losses, self.trailing_avg_window)
        smoothed_masked_seq_acc = self.moving_average(masked_seq_acc, self.trailing_avg_window)
        smoothed_unmasked_seq_acc = self.moving_average(unmasked_seq_acc, self.trailing_avg_window)

        smoothed_steps = steps[self.trailing_avg_window - 1:] if len(steps) >= self.trailing_avg_window else steps

        plt.figure(figsize=(16, 12))

        plt.subplot(2, 2, 1)
        if smoothed_training_losses and smoothed_validation_losses:
            plt.plot(smoothed_steps, smoothed_training_losses, label='Smoothed Training Loss')
            plt.plot(smoothed_steps, smoothed_validation_losses, label='Smoothed Validation Loss')
        else:
            plt.plot(steps, training_losses, label='Training Loss')
            plt.plot(steps, validation_losses, label='Validation Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Smoothed Training and Validation Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        if smoothed_masked_seq_acc and smoothed_unmasked_seq_acc:
            plt.plot(smoothed_steps, smoothed_masked_seq_acc, label='Smoothed Masked Seq Loss')
            plt.plot(smoothed_steps, smoothed_unmasked_seq_acc, label='Smoothed Unmasked Seq Loss')
        else:
            plt.plot(steps, masked_seq_acc, label='Masked Seq Loss')
            plt.plot(steps, unmasked_seq_acc, label='Unmasked Seq Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Masked and Unmasked Validation Loss')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(steps, training_losses, label='Raw Training Loss', alpha=0.7)
        plt.plot(steps, validation_losses, label='Raw Validation Loss', alpha=0.7)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Raw Training and Validation Loss')
        plt.legend()

        plt.tight_layout()

        if save_plot:
            plt.savefig(os.path.join(self.experiment_dir, 'training_validation_loss_plots.png'))
        else:
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--eval_interval', type=int, default=100, help='How often to run validation')
    parser.add_argument('--input_dim', type=int, default=513, help='Input dimension (frequency bins)')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for encoders')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers for encoders')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads for encoders')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--mlp_dim', type=int, default=128, help='MLP hidden dimension for encoders')
    parser.add_argument('--pred_hidden_dim', type=int, default=64, help='Hidden dimension for predictor')
    parser.add_argument('--pred_num_layers', type=int, default=2, help='Number of transformer layers for predictor')
    parser.add_argument('--pred_num_heads', type=int, default=2, help='Number of attention heads for predictor')
    parser.add_argument('--pred_mlp_ratio', type=float, default=4.0, help='MLP ratio for predictor')
    parser.add_argument('--max_seq_len', type=int, default=500, help='Maximum sequence length for predictor')
    parser.add_argument('--overfit_batch', action='store_true', 
                       help='Whether to overfit on a single batch (default: False)')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                       help='Ratio of timesteps to mask (default: 0.75)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output during training')
    args = parser.parse_args()

    root_exp_dir = "experiments"
    archive_dir = os.path.join(root_exp_dir, "archive")
    experiment_dir = os.path.join(root_exp_dir, args.name)
    
    # Create archive directory if it doesn't exist
    os.makedirs(archive_dir, exist_ok=True)

    # If experiment directory already exists and we're not continuing training
    if os.path.exists(experiment_dir):
        # Create timestamped name for archived experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_name = f"{args.name}_{timestamp}"
        archived_path = os.path.join(archive_dir, archived_name)
        
        # Move existing experiment to archive
        shutil.move(experiment_dir, archived_path)
        print(f"Archived existing experiment to: {archived_path}")

    # Create new experiment directory
    os.makedirs(experiment_dir, exist_ok=True)
    weights_dir = os.path.join(experiment_dir, 'saved_weights')

    os.makedirs(experiment_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dl_train = DataLoader(
        BirdJEPA_Dataset(data_dir=args.train_dir, segment_len=args.max_seq_len, verbose=False),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, 
                                          segment_length=args.max_seq_len, 
                                          mask_p=args.mask_ratio, 
                                          verbose=False)
    )

    dl_test = DataLoader(
        BirdJEPA_Dataset(data_dir=args.test_dir, segment_len=args.max_seq_len, verbose=False),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, 
                                          segment_length=args.max_seq_len, 
                                          mask_p=args.mask_ratio, 
                                          verbose=False)
    )

    # ensure input_dim matches D from dataset
    # set input_dim=... (e.g., 256)
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
        pred_mlp_ratio=args.pred_mlp_ratio,
        max_seq_len=args.max_seq_len
    ).to(device)
    opt = torch.optim.Adam(list(model.context_encoder.parameters()) + 
                          list(model.predictor.parameters()) + 
                          list(model.decoder.parameters()), lr=args.lr)

    # Create config dictionary
    config = {
        # Data parameters
        "input_dim": args.input_dim,
        "max_seq_len": args.max_seq_len,
        "mask_ratio": args.mask_ratio,
        
        # Model architecture
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "mlp_dim": args.mlp_dim,
        
        # Predictor parameters
        "pred_hidden_dim": args.pred_hidden_dim,
        "pred_num_layers": args.pred_num_layers,
        "pred_num_heads": args.pred_num_heads,
        "pred_mlp_ratio": args.pred_mlp_ratio,
        
        # Training parameters
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_steps": args.steps,
        "eval_interval": args.eval_interval,
        
        # Paths and names
        "experiment_name": args.name,
        "train_dir": args.train_dir,
        "test_dir": args.test_dir,
        
        # Additional info
        "device": str(device),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Save config
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    trainer = ModelTrainer(model=model, 
                          train_loader=dl_train, 
                          test_loader=dl_test, 
                          optimizer=opt, 
                          device=device, 
                          max_steps=args.steps, 
                          eval_interval=args.eval_interval,
                          save_interval=500, 
                          weights_save_dir=weights_dir,
                          experiment_dir=experiment_dir,
                          verbose=args.verbose,
                          overfit_single_batch=args.overfit_batch)

    trainer.train()
    trainer.plot_results(save_plot=True)
