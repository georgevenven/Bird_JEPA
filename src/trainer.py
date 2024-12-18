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
                 max_steps=10000, eval_interval=500, save_interval=1000, 
                 weights_save_dir='saved_weights', overfit_on_batch=False, experiment_dir=None, 
                 early_stopping=True, patience=8, trailing_avg_window=1000, verbose=False):

        self.overfit_on_batch = overfit_on_batch
        self.fixed_batch = None
        self.model = model
        self.train_iter = train_loader
        self.test_iter = test_loader
        self.optimizer = optimizer
        self.device = device
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.scheduler = StepLR(optimizer, step_size=10000, gamma=1)
        self.early_stopping = early_stopping
        self.patience = patience
        self.trailing_avg_window = trailing_avg_window
        self.save_interval = save_interval
        self.weights_save_dir = weights_save_dir
        self.experiment_dir = experiment_dir
        self.count_reinit = 0
        self.verbose = verbose

        if not os.path.exists(self.weights_save_dir):
            os.makedirs(self.weights_save_dir)

        self.predictions_subfolder_path = os.path.join(experiment_dir, "predictions") if experiment_dir else "predictions"
        if not os.path.exists(self.predictions_subfolder_path):
            os.makedirs(self.predictions_subfolder_path)

    def embedding_variance(self, embeddings):
        """Compute the mean variance of embeddings across the batch."""
        return embeddings.var(dim=0).mean().item()

    def sum_squared_weights(self):
        return sum(torch.sum(p ** 2) for p in self.model.parameters())

    def save_model(self, step, training_stats):
        filename = f"model_step_{step}.pth"
        filepath = os.path.join(self.weights_save_dir, filename)
        torch.save(self.model.state_dict(), filepath)

        stats_filename = "training_statistics.json"
        stats_filepath = os.path.join(self.experiment_dir, stats_filename)
        with open(stats_filepath, 'w') as json_file:
            json.dump(training_stats, json_file, indent=4)

    def create_large_canvas(self, context_outputs, target_outputs=None, image_idx=0, spec_shape=None, spec=None, target_spec=None):
        # Determine number of rows and columns
        context_layers = context_outputs["layer_outputs"].shape[0]  # (#layers,B,T,H)
        total_rows = context_layers + 1  # +1 for input spectrograms
        num_cols = 2 if target_outputs is not None else 1

        fig, axes = plt.subplots(total_rows, num_cols, figsize=(20, 5 * total_rows))
        if total_rows == 1 and num_cols == 1:
            axes = np.array([[axes]])
        elif total_rows == 1 or num_cols == 1:
            axes = axes.reshape(total_rows, num_cols)

        y_axis_limit = spec_shape[1] if spec_shape else None

        # Plot input spectrograms
        if spec_shape is not None:
            axes[0,0].imshow(spec.cpu()[image_idx].numpy(), aspect='auto', origin='lower')
            axes[0,0].set_title("Input to Context Encoder")
            axes[0,1].imshow(target_spec.cpu()[image_idx].numpy(), aspect='auto', origin='lower')
            axes[0,1].set_title("Input to Target Encoder")

        # Plot encoder outputs
        for i in range(context_layers):
            # Context encoder outputs
            ax = axes[i+1,0]
            image_data = context_outputs["layer_outputs"][i, image_idx].cpu().detach().numpy()
            ax.imshow(image_data.T, aspect='auto', origin='lower')
            if y_axis_limit:
                ax.set_ylim(bottom=0, top=y_axis_limit)
            ax.set_title(f"Context Encoder Layer {i}")

            # Target encoder outputs
            if target_outputs is not None:
                ax = axes[i+1,1]
                image_data = target_outputs["layer_outputs"][i, image_idx].cpu().detach().numpy()
                ax.imshow(image_data.T, aspect='auto', origin='lower')
                if y_axis_limit:
                    ax.set_ylim(bottom=0, top=y_axis_limit)
                ax.set_title(f"Target Encoder Layer {i}")

        plt.subplots_adjust(hspace=0.3)
        plt.tight_layout()

    def visualize_mse(self, output, mask, spec, step, full_spectrogram=None):
        # output,spec,mask: (B,D,T)
        output_np = output.cpu().numpy()
        spec_np = spec.cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        # Create error heatmap
        error_map = ((output - spec)**2 * mask.float())[0].cpu().numpy()  # (D,T)

        # Increase number of subplots
        fig, axs = plt.subplots(2, 2, figsize=(30, 20))
        axs = axs.ravel()

        x_label = 'Time Bins'
        y_label = 'Frequency Bins'
        plt.subplots_adjust(hspace=0.33, wspace=0.2)
        for ax in axs:
            ax.tick_params(axis='both', which='major', labelsize=25, length=6, width=2)
            ax.set_xlabel(x_label, fontsize=25)
            ax.set_ylabel(y_label, fontsize=25)

        # Plot 1: Original unmasked spectrogram
        if full_spectrogram is not None:
            axs[0].imshow(full_spectrogram[0].cpu().numpy(), aspect='auto', origin='lower')
            axs[0].set_title('Original Spectrogram (No Mask)', fontsize=35, pad=20)

        # Plot 2: Masked input
        axs[1].imshow(spec_np[0], aspect='auto', origin='lower')
        axs[1].set_title('Masked Input Spectrogram', fontsize=35, pad=20)
        self._add_mask_overlay(axs[1], mask_np[0])

        # Plot 3: Model prediction
        axs[2].imshow(output_np[0], aspect='auto', origin='lower')
        axs[2].set_title('Model Prediction', fontsize=35, pad=20)
        self._add_mask_overlay(axs[2], mask_np[0])

        # Plot 4: Error heatmap
        im = axs[3].imshow(error_map, aspect='auto', origin='lower', cmap='hot')
        axs[3].set_title('Prediction Error Heatmap', fontsize=35, pad=20)
        plt.colorbar(im, ax=axs[3])

        plt.savefig(os.path.join(self.predictions_subfolder_path, f'MSE_Visualization_{step}.png'), format="png")
        plt.close(fig)

    def _add_mask_overlay(self, axis, mask):
        # mask: (D,T)
        # highlight masked time steps
        masked_tokens = (mask != 0).any(axis=0) # bool for each time step
        y_min, y_max = axis.get_ylim()
        mask_bar_position = y_max - 15
        for idx,m in enumerate(masked_tokens):
            if m:
                axis.add_patch(plt.Rectangle((idx, mask_bar_position), 1, 15, 
                                             edgecolor='none', facecolor='red'))

    def visualize_masked_predictions(self, step, spec, output, mask, all_outputs, full_spectrogram=None):
        self.model.eval()
        with torch.no_grad():
            # Get target encoder outputs using target_spectrogram
            target_outputs = None
            if hasattr(self.model, 'target_encoder'):
                # Get target_spectrogram from the data loader
                _, target_spectrogram, context_spectrogram, _, _, _ = next(iter(self.test_iter))
                target_spectrogram = target_spectrogram.to(self.device)
                _, target_intermediate = self.model.target_encoder(target_spectrogram)
                target_outputs = {"layer_outputs": torch.stack(target_intermediate, dim=0)}

            # Visualize MSE and predictions
            self.visualize_mse(output=output, mask=mask, spec=spec, step=step, 
                              full_spectrogram=full_spectrogram)
            
            # Pass spec to create_large_canvas
            self.create_large_canvas(all_outputs, target_outputs, image_idx=0, 
                                   spec_shape=spec.shape, spec=spec, target_spec=target_spectrogram)
            plt.savefig(os.path.join(self.predictions_subfolder_path, 
                       f'Intermediate_Outputs_{step}.png'), format="png")
            plt.close()

    def validate_model(self, step, spec, vocalization):
        self.model.eval()
        with torch.no_grad():
            if self.verbose:
                print(f"\nValidation Input Shapes:")
                print(f"spec: {spec.shape}")
                print(f"vocalization: {vocalization.shape}")

            spec = spec.to(self.device)
            decoded_pred, mask, target_spectrogram, intermediate_outputs = self.model.train_forward(spec)

            if self.verbose:
                print(f"\nValidation Output Shapes:")
                print(f"decoded_pred: {decoded_pred.shape}")
                print(f"mask: {mask.shape}")
                print(f"target_spectrogram: {target_spectrogram.shape}")
                print(f"intermediate_outputs['layer_outputs']: {intermediate_outputs['layer_outputs'].shape}")

            if step % self.eval_interval == 0 or step == 0:
                self.visualize_masked_predictions(step, spec=spec, output=decoded_pred, mask=mask, all_outputs=intermediate_outputs)
            val_loss, masked_seq_acc, unmasked_seq_acc = self.model.mse_loss(predictions=decoded_pred, spec=target_spectrogram, mask=mask, vocalization=vocalization, intermediate_layers=intermediate_outputs)
            return val_loss.item(), masked_seq_acc.item(), unmasked_seq_acc.item()

    def moving_average(self, values, window):
        if len(values) < window:
            return []
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
        return sma.tolist()

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

        while step < self.max_steps:
            try:
                train_batch = next(train_iter)
                val_batch = next(test_iter)
            except StopIteration:
                train_iter = iter(self.train_iter)
                test_iter = iter(self.test_iter)
                continue

            full_spectrogram, target_spectrogram, context_spectrogram, ground_truth_labels, vocalization, file_names = train_batch
            
            if self.verbose and step % self.eval_interval == 0:
                # the shape is (B,F,T) for spectograms 
                # F is the number of frequency bins
                # T is the number of time bins
                # B is the batch size
                # the shape is (B,T) for labels
                
                print(f"\nStep {step} Input Shapes:")
                print(f"full_spectrogram: {full_spectrogram.shape}")
                print(f"target_spectrogram: {target_spectrogram.shape}")
                print(f"context_spectrogram: {context_spectrogram.shape}")
                print(f"ground_truth_labels: {ground_truth_labels.shape}")
                print(f"vocalization: {vocalization.shape}")
            
            context_spectrogram = context_spectrogram.to(self.device)
            target_spectrogram = target_spectrogram.to(self.device)

            self.model.train()
            with autocast():
                decoded_pred, mask, t_spect, intermediate_outputs = self.model.train_forward(context_spectrogram)
                
                if self.verbose and step % self.eval_interval == 0:
                    print(f"\nStep {step} Output Shapes:")
                    print(f"decoded_pred: {decoded_pred.shape}")
                    print(f"mask: {mask.shape}")
                    print(f"t_spect: {t_spect.shape}")
                    print(f"intermediate_outputs['layer_outputs']: {intermediate_outputs['layer_outputs'].shape}")

                loss, masked_seq_acc, unmasked_seq_acc = self.model.mse_loss(
                    predictions=decoded_pred, spec=t_spect, mask=mask, 
                    vocalization=vocalization, intermediate_layers=intermediate_outputs)

            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.scheduler.step()

            raw_loss_list.append(loss.item())

            val_full_spect, val_target_spect, val_context_spect, val_gt_labels, val_vocalization, val_file_names = val_batch
            val_context_spect = val_context_spect.to(self.device)
            val_vocalization = val_vocalization.to(self.device)

            with autocast():
                val_loss, avg_masked_seq_acc, avg_unmasked_seq_acc = self.validate_model(step, spec=val_context_spect, vocalization=val_vocalization)

            raw_val_loss_list.append(val_loss)
            raw_masked_seq_acc_list.append(avg_masked_seq_acc)
            raw_unmasked_seq_acc_list.append(avg_unmasked_seq_acc)

            if step % self.eval_interval == 0 or step == 0:
                # Calculate variances here
                with torch.no_grad():
                    context_repr, _ = self.model.context_encoder(context_spectrogram)
                    target_repr, _ = self.model.target_encoder(target_spectrogram)
                    context_var = self.embedding_variance(context_repr)
                    target_var = self.embedding_variance(target_repr)

                if len(raw_loss_list) >= self.trailing_avg_window:
                    smoothed_training_loss = self.moving_average(raw_loss_list, self.trailing_avg_window)[-1]
                    smoothed_val_loss = self.moving_average(raw_val_loss_list, self.trailing_avg_window)[-1]
                    smoothed_masked_seq_acc = self.moving_average(raw_masked_seq_acc_list, self.trailing_avg_window)[-1]
                    smoothed_unmasked_seq_acc = self.moving_average(raw_unmasked_seq_acc_list, self.trailing_avg_window)[-1]
                else:
                    smoothed_training_loss = loss.item()
                    smoothed_val_loss = val_loss
                    smoothed_masked_seq_acc = avg_masked_seq_acc
                    smoothed_unmasked_seq_acc = avg_unmasked_seq_acc

                print(f'Step [{step}/{self.max_steps}], '
                      f'Smoothed Training Loss: {smoothed_training_loss:.4f}, '
                      f'Smoothed Validation Loss: {smoothed_val_loss:.4f}, '
                      f'Smoothed Masked Seq Loss: {smoothed_masked_seq_acc:.4f}, '
                      f'Smoothed Unmasked Seq Loss: {smoothed_unmasked_seq_acc:.4f}, '
                      f'Context Var: {context_var:.6f}, Target Var: {target_var:.6f}')

                if context_var < 1e-5 or target_var < 1e-5:
                    print(f"Warning: Potential collapse detected at step {step} (low variance).")

                if smoothed_val_loss < best_val_loss:
                    best_val_loss = smoothed_val_loss
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1

                if self.early_stopping and steps_since_improvement >= self.patience:
                    print(f"Early stopping triggered at step {step}. No improvement for {self.patience} intervals.")
                    break

            if step % self.save_interval == 0 or step == self.max_steps - 1:
                current_training_stats = {
                    'step': step,
                    'training_loss': raw_loss_list,
                    'masked_seq_acc': raw_masked_seq_acc_list,
                    'unmasked_seq_acc': raw_unmasked_seq_acc_list,
                    'validation_loss': raw_val_loss_list,
                    'steps_since_improvement': steps_since_improvement,
                    'best_val_loss': best_val_loss
                }
                self.save_model(step, current_training_stats)

            step += 1

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
        BirdJEPA_Dataset(data_dir=args.train_dir, segment_len=500, verbose=False),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, segment_length=500, mask_p=0.5, verbose=False)
    )

    dl_test = DataLoader(
        BirdJEPA_Dataset(data_dir=args.test_dir, segment_len=500, verbose=False),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, segment_length=500, mask_p=0.5, verbose=False)
    )

    # ensure input_dim matches D from dataset
    # set input_dim=... (e.g., 256)
    model = BirdJEPA(input_dim=513, hidden_dim=64).to(device)
    opt = torch.optim.Adam(list(model.context_encoder.parameters()) + list(model.predictor.parameters()) + list(model.decoder.parameters()), lr=args.lr)

    trainer = ModelTrainer(model=model, 
                          train_loader=dl_train, 
                          test_loader=dl_test, 
                          optimizer=opt, 
                          device=device, 
                          max_steps=args.steps, 
                          eval_interval=100, 
                          save_interval=500, 
                          weights_save_dir=weights_dir,
                          experiment_dir=experiment_dir,
                          verbose=False)

    trainer.train()
    trainer.plot_results(save_plot=True)
