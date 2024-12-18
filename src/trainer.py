# trainer.py
import os
import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import json
from torch.cuda.amp import autocast, GradScaler
from birdjepa import BirdJEPA
from data_class import BirdJEPA_Dataset, collate_fn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse

class ModelTrainer:
    def __init__(self, model, train_loader, test_loader, optimizer, device, 
                 max_steps=10000, eval_interval=500, save_interval=1000, 
                 weights_save_dir='saved_weights', overfit_on_batch=False, experiment_dir=None, 
                 early_stopping=True, patience=8, trailing_avg_window=1000):

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

        if not os.path.exists(self.weights_save_dir):
            os.makedirs(self.weights_save_dir)

        self.predictions_subfolder_path = os.path.join(experiment_dir, "predictions") if experiment_dir else "predictions"
        if not os.path.exists(self.predictions_subfolder_path):
            os.makedirs(self.predictions_subfolder_path)

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

    def create_large_canvas(self, intermediate_outputs, image_idx=0, spec_shape=None):
        layer_outputs = intermediate_outputs["layer_outputs"] # (#layers,B,T,H)
        num_layers = layer_outputs.shape[0]

        fig, axes = plt.subplots(num_layers, 1, figsize=(10, 3 * num_layers))
        if num_layers == 1:
            axes = [axes]

        y_axis_limit = spec_shape[1] if spec_shape else None

        for i, ax in enumerate(axes):
            image_data = layer_outputs[i, image_idx].cpu().detach().numpy() # (T,H)
            ax.imshow(image_data.T, aspect='auto', origin='lower')
            if y_axis_limit:
                ax.set_ylim(bottom=0, top=y_axis_limit)
            ax.set_aspect('auto')
            ax.set_title(f"Layer {i} output")
            ax.axis('off')

        plt.tight_layout()

    def visualize_mse(self, output, mask, spec, step):
        # output,spec,mask: (B,D,T)
        output_np = output.cpu().numpy()
        spec_np = spec.cpu().numpy()
        mask_np = mask.cpu().numpy()

        fig, axs = plt.subplots(2, 1, figsize=(30, 20))
        axs = axs.ravel()

        x_label = 'Time Bins'
        y_label = 'Frequency Bins'
        plt.subplots_adjust(hspace=0.33)
        for ax in axs:
            ax.tick_params(axis='both', which='major', labelsize=25, length=6, width=2)
            ax.set_xlabel(x_label, fontsize=25)
            ax.set_ylabel(y_label, fontsize=25)

        axs[0].imshow(spec_np[0], aspect='auto', origin='lower')
        axs[0].set_title('Original Spectrogram with Mask', fontsize=35, pad=20)
        self._add_mask_overlay(axs[0], mask_np[0])

        axs[1].imshow(output_np[0], aspect='auto', origin='lower')
        axs[1].set_title('Model Prediction with Mask', fontsize=35, pad=20)
        self._add_mask_overlay(axs[1], mask_np[0])

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

    def visualize_masked_predictions(self, step, spec, output, mask, all_outputs):
        self.model.eval()
        with torch.no_grad():
            self.visualize_mse(output=output, mask=mask, spec=spec, step=step)
            self.create_large_canvas(all_outputs, image_idx=0, spec_shape=spec.shape)
            plt.savefig(os.path.join(self.predictions_subfolder_path, f'Intermediate_Outputs_{step}.png'), format="png")
            plt.close()

    def validate_model(self, step, spec, vocalization):
        self.model.eval()
        with torch.no_grad():
            spec = spec.to(self.device)
            decoded_pred, mask, target_spectrogram, intermediate_outputs = self.model.train_forward(spec)
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
            context_spectrogram = context_spectrogram.to(self.device)
            target_spectrogram = target_spectrogram.to(self.device)

            self.model.train()
            with autocast():
                decoded_pred, mask, t_spect, intermediate_outputs = self.model.train_forward(context_spectrogram)
                loss, masked_seq_acc, unmasked_seq_acc = self.model.mse_loss(predictions=decoded_pred, spec=t_spect, mask=mask, vocalization=vocalization, intermediate_layers=intermediate_outputs)

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
                      f'Smoothed Unmasked Seq Loss: {smoothed_unmasked_seq_acc:.4f}')

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
    experiment_dir = os.path.join(root_exp_dir, args.name)
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
    model = BirdJEPA(input_dim=256, hidden_dim=64).to(device)
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
                           experiment_dir=experiment_dir)

    trainer.train()
    trainer.plot_results(save_plot=True)
