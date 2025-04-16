import os
import pandas as pd
import torch
import torch.nn as nn
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from datetime import datetime
import pandas.api.types
import sklearn.metrics
from sklearn.metrics import roc_auc_score
import shutil
from tqdm import tqdm
import csv 
import sys
# func torch
import torch.nn.functional as F
import onnxruntime as ort
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from torch.utils.data import SequentialSampler

# Import from project modules
from model import BirdJEPA
from utils import load_model
from spectrogram_generator import WavtoSpec
from data_class import BirdJEPA_Dataset, collate_fn as data_class_collate_fn

# ParticipantVisibleError for Kaggle metrics
class ParticipantVisibleError(Exception):
    pass

# Kaggle evaluation function
def kaggle_score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Version of macro-averaged ROC-AUC score that ignores all classes that have no true positive labels.
    '''
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if not pandas.api.types.is_numeric_dtype(submission.values):
        bad_dtypes = {x: submission[x].dtype for x in submission.columns if not pandas.api.types.is_numeric_dtype(submission[x])}
        raise ParticipantVisibleError(f'Invalid submission data types found: {bad_dtypes}')

    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)
    assert len(scored_columns) > 0

    try:
        return roc_auc_score(solution[scored_columns].values, submission[scored_columns].values, average='macro')
    except Exception as e:
        # Handle potential errors in scoring
        print(f"Error in calculating score: {e}")
        return 0.0

# Safe implementation for calculating ROC AUC score (equivalent to kaggle_metric_utilities.safe_call_score)
def safe_roc_auc_score(y_true, y_pred, average='macro'):
    """Safely calculate ROC AUC score, handling edge cases."""
    try:
        return roc_auc_score(y_true, y_pred, average=average)
    except ValueError as e:
        # Handle case where a class might not have both positive and negative samples
        print(f"Warning in ROC AUC calculation: {e}")
        # Fall back to calculating for classes that can be scored
        if average == 'macro':
            # Calculate per-class scores where possible
            classes = np.unique(np.where(y_true == 1)[1])
            if len(classes) == 0:
                return 0.0
            scores = []
            for c in classes:
                try:
                    class_score = roc_auc_score(y_true[:, c], y_pred[:, c])
                    scores.append(class_score)
                except ValueError:
                    # Skip classes that cause errors
                    pass
            return np.mean(scores) if scores else 0.0
        return 0.0

# --- modified BirdCLEFDataWrapper to allow custom collate_fn and shuffle ---
class BirdCLEFDataWrapper:
    def __init__(self, csv_path, data_dir, context_length, batch_size, infinite_dataset=True, collate_fn=None, shuffle=True):
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.context_length = context_length
        self.infinite_dataset = infinite_dataset
        
        # DEBUG: Print key initialization parameters
        print(f"DEBUG: BirdCLEFDataWrapper init with data_dir={data_dir}, infinite_dataset={infinite_dataset}, shuffle={shuffle}")
        
        # Load CSV file into memory 
        self.csv_data = pd.read_csv(self.csv_path)
        
        # Extract primary_label and filename columns
        self.primary_labels = self.csv_data['primary_label'].values
        self.filenames = self.csv_data['filename'].values if 'filename' in self.csv_data.columns else self.csv_data['file_name'].values
        
        # Get unique classes and create label mapping
        self.unique_labels = sorted(self.csv_data['primary_label'].unique())
        self.num_classes = len(self.unique_labels)
        self.label_to_idx = {label: i for i, label in enumerate(self.unique_labels)}
        
        # Create a mapping from identifier to label
        self.identifier_to_label = {}
        pattern = re.compile(r'(XC\d+|iNat\d+|CSA\d+)')
        
        for i, row in self.csv_data.iterrows():
            filename = row['filename'] if 'filename' in self.csv_data.columns else row['file_name']
            # Remove directory part if present
            if '/' in filename:
                filename = filename.split('/')[-1]
            
            # Extract the identifier (XC number or iNat number)
            match = pattern.search(filename)
            if match:
                identifier = match.group(1)
                self.identifier_to_label[identifier] = row['primary_label']
        
        print(f"Created mapping for {len(self.identifier_to_label)} files")
        
        # Initialize the dataset
        self.init_dataset(context_length, collate_fn, shuffle)
        
        # DEBUG: Track which files have been processed
        self.processed_files = []
        self.batch_counter = 0

    def get_label_for_file(self, filename):
        """Get the primary label for a given filename based on XC/iNat identifier"""
        # Extract the identifier from the filename
        pattern = re.compile(r'(XC\d+|iNat\d+|CSA\d+)')
        
        # Extract the identifier from the filename
        match = pattern.search(filename)
        if match:
            identifier = match.group(1)
            if identifier in self.identifier_to_label:
                label = self.identifier_to_label[identifier]
                label_idx = self.label_to_idx[label]
                return label, label_idx
        
        print(f"Warning: No label found for {filename}")
        return None, -1

    def init_dataset(self, context_length, collate_fn, shuffle):
        print(f"Initializing dataset from {self.data_dir}")
        
        # DEBUG: List actual files in directory
        try:
            dir_files = os.listdir(self.data_dir)
            print(f"DEBUG: Files in {self.data_dir}: {dir_files}")
        except Exception as e:
            print(f"DEBUG: Error listing directory: {e}")
        
        self.dataset = BirdJEPA_Dataset(data_dir=self.data_dir, segment_len=context_length, infinite_dataset=self.infinite_dataset)
        
        # DEBUG: Print actual dataset file paths
        print(f"DEBUG: Dataset file paths: {self.dataset.file_paths}")
        
        if collate_fn is None:
            collate_fn = data_class_collate_fn
        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn if self.infinite_dataset else None,
            sampler=SequentialSampler(self.dataset)  # Force sequential access
        )
        
        self.dataloader_iter = iter(self.dataloader)
        print(f"Dataset initialized with {len(self.dataset)} samples")
        
        # DEBUG: Track iteration state
        self.iteration_count = 0
        self.reset_count = 0
    
    def next_batch(self):
        # DEBUG: Track batch requests
        self.batch_counter += 1
        print(f"\nDEBUG: next_batch called ({self.batch_counter}), iteration_count={self.iteration_count}, reset_count={self.reset_count}")
        
        try:
            self.iteration_count += 1
            batch = next(self.dataloader_iter)
            
            # DEBUG: Print batch structure for diagnostics
            print(f"DEBUG: Batch type: {type(batch)}, structure: {type(batch) if not isinstance(batch, (tuple, list)) else [type(item) for item in batch]}")
            
            # If we want to extract labels for the files in this batch
            if isinstance(batch, tuple) and len(batch) >= 6:
                filenames = batch[5]
                batch_labels = []
                batch_label_indices = []
                
                # DEBUG: Track filenames
                print(f"DEBUG: Processing batch with filenames: {filenames}")
                
                for filename in filenames:
                    # Track processed files
                    self.processed_files.append(filename)
                    
                    label, label_idx = self.get_label_for_file(filename)
                    batch_labels.append(label)
                    batch_label_indices.append(label_idx)
                
                # Return batch along with extracted labels
                return batch, batch_labels, batch_label_indices
            
            # this is sketchy, but when analyzing, we don't have a collate fn, so we return segment, segment_labels, os.path.basename(file_path) from dataclass
            if isinstance(batch, list) and len(batch) >= 3:
                filenames = batch[2]
                
                # DEBUG: Track filenames
                print(f"DEBUG: Processing analysis batch with filenames: {filenames}")
                
                batch_labels = []
                batch_label_indices = []
                
                for filename in filenames:
                    # Track processed files
                    self.processed_files.append(filename)
                    
                    label, label_idx = self.get_label_for_file(filename)
                    batch_labels.append(label)
                    batch_label_indices.append(label_idx)

                return batch[0], batch_labels, batch_label_indices, filenames

            # If no filenames to process, just return the batch with empty label lists
            print("DEBUG: No filenames found in batch")
            return batch, [], []
        except StopIteration:
            # Handle StopIteration based on whether we're using an infinite dataset
            if self.infinite_dataset:
                # Reset iterator if we've gone through the entire dataset
                self.dataloader_iter = iter(self.dataloader)
                self.reset_count += 1
                print(f"DEBUG: StopIteration reached, resetting dataloader_iter (reset #{self.reset_count})")
                
                # DEBUG: Report on processed files
                if self.processed_files:
                    unique_files = set(self.processed_files)
                    print(f"DEBUG: Processed {len(self.processed_files)} total files, {len(unique_files)} unique")
                    print(f"DEBUG: Unique files processed: {sorted(unique_files)}")
                    
                    # Count occurrences of each file
                    from collections import Counter
                    file_counts = Counter(self.processed_files)
                    print(f"DEBUG: File counts: {file_counts}")
                
                return self.next_batch()
            else:
                # For finite datasets, we should not reset - just raise the exception
                # This allows the caller to know we've processed all files
                print(f"DEBUG: End of dataset reached after {self.iteration_count} iterations")
                raise

class Classifier(nn.Module):
    def __init__(self, context_length, num_classes, hidden_dim=64, pool_type='mean'):
        super(Classifier, self).__init__()
        self.pool_type = pool_type  # 'mean' or 'max'
        
        # MLP after pooling (much smaller now)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
            # Remove sigmoid/softmax - BCEWithLogitsLoss will apply sigmoid internally
        )    

    def forward(self, x):
        # x shape: [batch_size, context_length, hidden_dim]
        
        # Apply global pooling across temporal dimension
        if self.pool_type == 'mean':
            # Mean pooling over the temporal dimension
            pooled = torch.mean(x, dim=1)  # [batch_size, hidden_dim]
        elif self.pool_type == 'max':
            # Max pooling over the temporal dimension
            pooled, _ = torch.max(x, dim=1)  # [batch_size, hidden_dim]
        
        # Pass through MLP
        return self.mlp(pooled)

class Model(nn.Module):
    def __init__(self, model_path=None, context_length=500, num_classes=1, pool_type='mean'):
        super(Model, self).__init__()
        self.model_path = model_path
        self.context_length = context_length
        self.num_classes = num_classes
        self.encoder = None
        self.classifier = Classifier(context_length, num_classes, hidden_dim=64, pool_type=pool_type)
        
        # Load pretrained model if path is provided
        if model_path:
            print(f"Loading pretrained encoder from {model_path}")
            self.encoder = load_model(model_path, load_weights=True)
        else:
            print("No pretrained encoder path provided. Model will need to be initialized from a checkpoint.")
        
        print(f"Model initialized with context_length={context_length}, num_classes={num_classes}, pooling={pool_type}")
    
    def forward(self, x):
        # Get embeddings from encoder
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        if self.encoder is None:
            raise RuntimeError("Encoder not initialized. Please load a pretrained model or checkpoint.")

        embedding_repr = self.encoder.inference_forward(x)

        embedding_repr = embedding_repr[0]  # [batch_size, context_length, hidden_dim]
        # No need to flatten, pass directly to classifier
        outputs = self.classifier(embedding_repr)
        return outputs
        
    def load_from_checkpoint(self, checkpoint_path, strict=True):
        """
        Load model weights from a checkpoint file
        """
        print(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Check if model_config is available in the checkpoint
        if 'model_config' in checkpoint and self.encoder is None:
            config = checkpoint['model_config']
            if 'pretrained_model_path' in config and config['pretrained_model_path']:
                print(f"Initializing encoder from config: {config['pretrained_model_path']}")
                self.encoder = load_model(config['pretrained_model_path'], load_weights=True)
        
        # Load state dict with specified strictness
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        return checkpoint

class Trainer:
    def __init__(self, args):
        self.args = args
        
        # Initialize data wrappers first to get num_classes
        print("Initializing data wrappers...")
        self.train_wrapper = BirdCLEFDataWrapper(
            csv_path=self.args.train_csv, 
            data_dir=self.args.train_spec_dir,
            context_length=self.args.context_length,
            batch_size=self.args.batch_size
        )

        self.val_wrapper = BirdCLEFDataWrapper(
            csv_path=self.args.train_csv, 
            data_dir=self.args.val_spec_dir,
            context_length=self.args.context_length,
            batch_size=self.args.batch_size
        )
        
        # Now initialize model with proper num_classes
        self.model = Model(
            args.pretrained_model_path, 
            args.context_length,
            num_classes=self.train_wrapper.num_classes,
            pool_type=args.pool_type
        )
        
        # If a resume checkpoint is provided, load it
        if hasattr(args, 'resume_checkpoint') and args.resume_checkpoint:
            print(f"Resuming from checkpoint: {args.resume_checkpoint}")
            checkpoint = self.model.load_from_checkpoint(args.resume_checkpoint, strict=False)
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        
        # Set up optimizer and loss function
        if self.args.freeze_encoder:
            # Only train the classifier if freeze_encoder is enabled
            self.optimizer = torch.optim.Adam(
                self.model.classifier.parameters(), 
                lr=self.args.learning_rate
            )
            print("Freezing encoder: only classifier will be trained")
        else:
            # Train both encoder and classifier by default
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.args.learning_rate
            )
            print("Training both encoder and classifier")
        self.criterion = nn.BCEWithLogitsLoss()  # use BCEWithLogitsLoss for multi-label classification
        
        # Move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.classifier.to(self.device)
        if self.model.encoder:
            self.model.encoder.to(self.device)
            
        # Initialize loss and metric tracking
        self.train_losses = []
        self.val_losses = []
        self.train_losses_ema = []
        self.val_losses_ema = []
        self.eval_steps = []
        self.ema_alpha = args.ema_alpha if hasattr(args, 'ema_alpha') else 0.2
        
        self.val_roc_auc_scores = []
        self.val_roc_auc_scores_ema = []
        
        # new output directory handling: if a folder already exists with the same name, move it to /archive
        if args.output_dir:
            self.run_dir = Path(args.output_dir)
            if self.run_dir.exists():
                archive_dir = self.run_dir.parent / "archive"
                archive_dir.mkdir(parents=True, exist_ok=True)
                archive_path = archive_dir / f"{self.run_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                print(f"existing folder detected. archiving to {archive_path}")
                shutil.move(str(self.run_dir), str(archive_path))
            self.run_dir.mkdir(parents=True, exist_ok=True)
            # dump config details to config.json along with encoder config if available
            config_path = self.run_dir / "config.json"
            config_data = vars(self.args).copy()
            if self.model.encoder is not None:
                # assuming the encoder provides its relevant parameters via a get_config() method
                # otherwise, manually specify the attributes you want to save (e.g., num_layers, hidden_dim, etc.)
                if hasattr(self.model.encoder, "get_config"):
                    config_data['encoder_config'] = self.model.encoder.get_config()
                else:
                    config_data['encoder_config'] = {
                        "dummy_key": "dummy_value"  # replace with relevant encoder parameters
                    }
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=4)
        else:
            self.run_dir = None
        
        # get unique bird classes for ROC-AUC evaluation
        self.bird_classes = self.train_wrapper.unique_labels

    def calculate_ema(self, current_value, previous_ema=None):
        if previous_ema is None:
            return current_value
        return self.ema_alpha * current_value + (1 - self.ema_alpha) * previous_ema

    def train(self):
        print(f"Starting training loop for {self.args.max_steps} steps")
        running_loss = 0.0
        best_val_loss = float('inf')
        best_roc_auc = 0.0
        patience_counter = 0
        
        for step in range(self.args.max_steps):
            batch_tuple, batch_labels, batch_label_indices = self.train_wrapper.next_batch()
            if -1 in batch_label_indices:
                print("Skipping batch with missing labels")
                continue

            inputs = batch_tuple[0].to(self.device)
            batch_size = len(batch_label_indices)
            labels = torch.zeros(batch_size, self.train_wrapper.num_classes, device=self.device)
            for i, idx in enumerate(batch_label_indices):
                labels[i, idx] = 1.0

            self.optimizer.zero_grad()
            outputs = self.model.forward(inputs)
            loss = self.criterion(outputs, labels)            
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if (step + 1) % self.args.eval_interval == 0:
                avg_train_loss = running_loss / self.args.eval_interval
                val_loss, roc_auc_score_val = self.evaluate(model=self.model, criterion=self.criterion, val_wrapper=self.val_wrapper)

                self.train_losses.append(avg_train_loss)
                self.val_losses.append(val_loss)
                self.val_roc_auc_scores.append(roc_auc_score_val)
                self.eval_steps.append(step + 1)
                
                if len(self.train_losses_ema) == 0:
                    self.train_losses_ema.append(avg_train_loss)
                    self.val_losses_ema.append(val_loss)
                    self.val_roc_auc_scores_ema.append(roc_auc_score_val)
                else:
                    self.train_losses_ema.append(self.calculate_ema(avg_train_loss, self.train_losses_ema[-1]))
                    self.val_losses_ema.append(self.calculate_ema(val_loss, self.val_losses_ema[-1]))
                    self.val_roc_auc_scores_ema.append(self.calculate_ema(roc_auc_score_val, self.val_roc_auc_scores_ema[-1]))
                
                print(f"Step {step+1}/{self.args.max_steps}, Training Loss: {avg_train_loss:.4f} (EMA: {self.train_losses_ema[-1]:.4f}), "
                      f"Validation Loss: {val_loss:.4f} (EMA: {self.val_losses_ema[-1]:.4f}), "
                      f"ROC-AUC: {roc_auc_score_val:.4f} (EMA: {self.val_roc_auc_scores_ema[-1]:.4f})")
                
                if self.args.save_best_metric == 'loss' and val_loss < best_val_loss and self.run_dir:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model(step, "best_loss")
                elif self.args.save_best_metric == 'roc_auc' and roc_auc_score_val > best_roc_auc and self.run_dir:
                    best_roc_auc = roc_auc_score_val
                    patience_counter = 0
                    self.save_model(step, "best_roc_auc")
                else:
                    patience_counter += 1
                
                if (step + 1) % self.args.save_interval == 0 and self.run_dir:
                    self.save_model(step, "checkpoint")
                
                if patience_counter >= self.args.early_stopping_patience:
                    print(f"Early stopping triggered at step {step+1}")
                    break
                
                running_loss = 0.0
        
        self.save_loss_data()
        self.plot_losses()
        self.plot_roc_auc()
        self.model.eval()

    def save_model(self, step, prefix="checkpoint"):
        if not self.run_dir:
            return
        # save weight checkpoints to a 'weights' subfolder
        weights_dir = self.run_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        save_path = weights_dir / f"{prefix}_step_{step+1}.pt"
        
        # Save more comprehensive information about the model
        model_config = {
            'context_length': self.args.context_length,
            'num_classes': self.train_wrapper.num_classes,
            'pool_type': self.args.pool_type,
            'freeze_encoder': self.args.freeze_encoder,
            'pretrained_model_path': self.args.pretrained_model_path
        }
        
        torch.save({
            'step': step + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'val_loss': self.val_losses[-1] if self.val_losses else None,
            'model_config': model_config,
        }, save_path)
        print(f"Model saved to {save_path}")

    def save_loss_data(self):
        if not self.run_dir:
            return
        loss_data = {
            'steps': self.eval_steps,
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_loss_ema': self.train_losses_ema,
            'val_loss_ema': self.val_losses_ema,
            'val_roc_auc': self.val_roc_auc_scores,
            'val_roc_auc_ema': self.val_roc_auc_scores_ema
        }
        # save loss data as loss.json (instead of metrics_data.json)
        json_path = self.run_dir / 'loss.json'
        with open(json_path, 'w') as f:
            json.dump(loss_data, f, indent=4)
        
        csv_path = self.run_dir / 'loss_data.csv'
        df = pd.DataFrame(loss_data)
        df.to_csv(csv_path, index=False)
        
        print(f"Loss data saved to {json_path} and {csv_path}")

    # plot_losses and plot_roc_auc remain unchanged
    def plot_losses(self):
        if not self.run_dir:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.eval_steps, self.train_losses, 'b-', alpha=0.3, label='Train Loss')
        plt.plot(self.eval_steps, self.val_losses, 'r-', alpha=0.3, label='Validation Loss')
        plt.plot(self.eval_steps, self.train_losses_ema, 'b-', linewidth=2, label='Train Loss (EMA)')
        plt.plot(self.eval_steps, self.val_losses_ema, 'r-', linewidth=2, label='Validation Loss (EMA)')
        plt.title('Training and Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = self.run_dir / 'loss_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss plot saved to {plot_path}")

    def plot_roc_auc(self):
        if not self.run_dir or not self.val_roc_auc_scores:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.eval_steps, self.val_roc_auc_scores, 'g-', alpha=0.3, label='Validation ROC-AUC')
        plt.plot(self.eval_steps, self.val_roc_auc_scores_ema, 'g-', linewidth=2, label='Validation ROC-AUC (EMA)')
        plt.title('Validation ROC-AUC Score')
        plt.xlabel('Steps')
        plt.ylabel('ROC-AUC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0.5, 1.05)
        plot_path = self.run_dir / 'roc_auc_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC-AUC plot saved to {plot_path}")

    # evaluate remains unchanged
    def evaluate(self, model, criterion, val_wrapper):
        model.eval()
        total_loss = 0.0
        num_batches = 5
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for _ in range(num_batches):
                batch_tuple, batch_labels, batch_label_indices = val_wrapper.next_batch()
                if -1 in batch_label_indices:
                    continue
                inputs = batch_tuple[0].to(self.device)
                batch_size = len(batch_label_indices)
                labels = torch.zeros(batch_size, val_wrapper.num_classes, device=self.device)
                for i, idx in enumerate(batch_label_indices):
                    labels[i, idx] = 1.0
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                probs = torch.sigmoid(outputs)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(probs.cpu().numpy())
        avg_loss = total_loss / num_batches
        roc_auc = 0.0
        if all_labels and all_preds:
            try:
                all_labels = np.vstack(all_labels)
                all_preds = np.vstack(all_preds)
                solution_df = pd.DataFrame(all_labels, columns=self.bird_classes)
                submission_df = pd.DataFrame(all_preds, columns=self.bird_classes)
                row_ids = [f"row_{i}" for i in range(len(solution_df))]
                solution_df.insert(0, "row_id", row_ids)
                submission_df.insert(0, "row_id", row_ids)
                roc_auc = kaggle_score(solution_df.copy(), submission_df.copy(), "row_id")
            except Exception as e:
                print(f"Error calculating ROC-AUC: {e}")
        model.train()
        return avg_loss, roc_auc

class Analysis:
    def __init__(self, args):
        self.args = args
        self.run_dir = Path(args.output_dir)
        weights_dir = self.run_dir / "weights"
        checkpoint_files = list(weights_dir.glob("best_*_step_*.pt"))
        if not checkpoint_files:
            raise Exception("no best model checkpoint found for analysis")
        
        # Extract step numbers from filenames and sort by numeric value
        def get_step_number(file_path):
            match = re.search(r'step_(\d+)\.pt', str(file_path))
            if match:
                return int(match.group(1))  # Convert to integer for numeric sorting
            return 0
        
        # Sort by step number, so "5750" will be higher than "750"
        checkpoint_file = sorted(checkpoint_files, key=get_step_number)[-1]
        
        print(f"Loading model checkpoint from {checkpoint_file}")
        
        # Load number of classes from training csv
        self.num_classes = self.load_num_classes(args.train_csv)
        
        # Create the model first with minimal parameters
        self.model = Model(
            model_path=args.pretrained_model_path,  # This can be None if loading from checkpoint
            context_length=args.context_length,
            num_classes=self.num_classes,
            pool_type=args.pool_type
        )
        
        # Load the checkpoint weights into the model
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded model weights from checkpoint (step {checkpoint.get('step', 'unknown')})")
        
        self.model.eval()
        
        # Create validation dataloader with batch size 1, no shuffle, no custom collate_fn, and non-infinite dataset
        self.val_wrapper = BirdCLEFDataWrapper(
            csv_path=args.train_csv,
            data_dir=args.val_spec_dir,
            context_length=None,
            batch_size=1,
            infinite_dataset=False,
            collate_fn=None,
            shuffle=False
        )
    
    def load_num_classes(self, csv_path):
        df = pd.read_csv(csv_path)
        return len(sorted(df['primary_label'].unique()))

    def run_analysis(self):
        print("running analysis on validation data")
        all_labels = []
        all_preds = []
        num_samples = len(self.val_wrapper.dataset)  # Limit to 500 samples for debugging
        device = torch.device("cpu")
        self.model.to(device)
        
        for i in tqdm(range(num_samples), desc="Processing samples"):
            spec_tensor, label, label_idx, filenames = self.val_wrapper.next_batch()
            
            # Skip if no valid label
            if label_idx == -1:
                print(f"warning: missing label for sample {i}, skipping")
                continue
                
            # slice the spec tensor into 5 second segments, and shape it as batch x freq x timebins
            segments = []
            for j in range(0, spec_tensor.shape[-1], 1000):
                segment = spec_tensor[..., j:j+1000]
                if segment.shape[-1] < 1000:
                    # Pad with zeros to reach 1000
                    pad_size = 1000 - segment.shape[-1]
                    pad_tensor = torch.zeros(*segment.shape[:-1], pad_size, device=segment.device)
                    segment = torch.cat([segment, pad_tensor], dim=-1)
                segments.append(segment)
            
            if not segments:
                continue
                
            segments = torch.stack(segments, dim=0)
            segments = segments.squeeze(1)
            
            # Run the model on each segment
            with torch.no_grad():
                segments = segments.to(device)
                outputs = self.model.forward(segments)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                # Average predictions across all segments
                avg_prob = np.mean(probs, axis=0)
                
                # Create one-hot encoded true label
                true_label = np.zeros(self.num_classes)
                true_label[label_idx] = 1.0
                
                all_labels.append(true_label)
                all_preds.append(avg_prob)
                
        if len(all_labels) == 0:
            print("No valid samples for analysis")
            return
            
        # Calculate ROC-AUC score
        all_labels_arr = np.vstack(all_labels)
        all_preds_arr = np.vstack(all_preds)
        
        # Get bird class names
        bird_classes = sorted(pd.read_csv(self.args.train_csv)['primary_label'].unique())
        
        # Create dataframes for scoring
        solution_df = pd.DataFrame(all_labels_arr, columns=bird_classes)
        submission_df = pd.DataFrame(all_preds_arr, columns=bird_classes)
        
        # Add row_id column
        row_ids = [f"row_{i}" for i in range(solution_df.shape[0])]
        solution_df.insert(0, "row_id", row_ids)
        submission_df.insert(0, "row_id", row_ids)
        
        # Calculate score
        score = kaggle_score(solution_df.copy(), submission_df.copy(), "row_id")
        
        # Prepare results string
        result_str = f"Analysis results:\nNum samples: {len(all_labels)}\nROC-AUC: {score:.4f}\n"
        
        # Print results
        print(result_str)
        
        # Save results to file
        results_path = self.run_dir / "analysis_results.txt"
        with open(results_path, "w") as f:
            f.write(result_str)
            
        print(f"Analysis results saved to {results_path}")

class Inference:
    def __init__(self, args):
        self.args = args
        self.run_dir = Path(args.output_dir)
        
        # limit to 4 cores for inference (for AMD EPYC env)
        torch.set_num_threads(4)
        print("limiting pytorch to 4 cpu cores for inference")
        
        # setup logging to console.log file
        console_log_path = args.console_log
        self.log_file = open(console_log_path, "w")
        self.original_stdout = sys.stdout
        sys.stdout = self.Tee(self.original_stdout, self.log_file)
        print(f"starting inference mode, logging to {console_log_path}")
        
        # Load model and prepare for inference
        weights_dir = self.run_dir / "weights"
        checkpoint_files = list(weights_dir.glob("best_*_step_*.pt"))
        if not checkpoint_files:
            print("Warning: No best model checkpoint found for analysis. Proceeding with random predictions.")
            self.num_classes = self.load_num_classes(args.train_csv)
            self.model = None  # Set model to None to indicate no model is loaded
            self.use_random_predictions = True
            
            # Skip ONNX export and runtime initialization when no model is available
            print("Skipping ONNX export since no model is available. Will use random predictions.")
            self.ort_session = None
        else:
            # sort checkpoints by step number and pick the latest
            def get_step_number(file_path):
                match = re.search(r'step_(\d+)\.pt', str(file_path))
                return int(match.group(1)) if match else 0

            checkpoint_file = sorted(checkpoint_files, key=get_step_number)[-1]
            print(f"loading model checkpoint from {checkpoint_file}")

            # load number of classes from training csv
            self.num_classes = self.load_num_classes(args.train_csv)

            # create model instance using minimal parameters
            self.model = Model(
                model_path=args.pretrained_model_path,  # can be None if loading from checkpoint
                context_length=args.context_length,
                num_classes=self.num_classes,
                pool_type=args.pool_type
            )

            # force cpu usage by loading checkpoint with map_location
            print("loading checkpoint for ONNX export")
            checkpoint = torch.load(checkpoint_file, map_location="cpu")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"successfully loaded model weights from checkpoint (step {checkpoint.get('step', 'unknown')})")
            self.model.eval()
            
            self.use_random_predictions = False
            
            # Export the model to ONNX format
            print("Exporting model to ONNX format")
            onnx_path = Path(self.args.onnx_model_path) if self.args.onnx_model_path else self.run_dir / "model.onnx"
            self.export_to_onnx(self.model, onnx_path)
            
            # Initialize ONNX runtime session
            print(f"Initializing ONNX runtime session from {onnx_path}")
            # Configure ONNX runtime session options for optimal CPU performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4   # set threads to match core count
            sess_options.inter_op_num_threads = 4

            # Enable parallel execution mode
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            
            # Enable profiling
            sess_options.enable_profiling = True

            self.ort_session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=['CPUExecutionProvider'])
            print("ONNX runtime session initialized with optimized CPU settings")
        
        # create validation dataloader with batch size 1, no shuffle, non-infinite dataset
        self.val_wrapper = BirdCLEFDataWrapper(
            csv_path=args.train_csv,
            data_dir=args.val_spec_dir,
            context_length=None,
            batch_size=1,
            infinite_dataset=False,
            collate_fn=None,
            shuffle=False
        )

    def export_to_onnx(self, model, onnx_path):
        """Export PyTorch model to ONNX format."""
        # Create dummy input of shape [batch_size, freq, time]
        # Assuming freq dimension is 64, time dimension is 1000
        dummy_input = torch.randn(1, 513, 1000, dtype=torch.float32)
        
        # Export model to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            do_constant_folding=True,
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"Exported ONNX model to {onnx_path}")
        
        # Quantize the model to INT8
        quantized_path = str(onnx_path).replace(".onnx", "_quantized.onnx")
        try:
            print(f"Applying INT8 quantization to model...")
            quantize_dynamic(str(onnx_path), quantized_path, weight_type=QuantType.QInt8)
            print(f"Quantized model saved to {quantized_path}")
            
            # Use the quantized model instead of the original one
            if os.path.exists(quantized_path):
                onnx_path = quantized_path
                print(f"Using quantized model for inference")
            else:
                print(f"Quantization failed, using original model")
        except Exception as e:
            print(f"Quantization error: {e}")
            print(f"Continuing with original model")

    # tee class to duplicate stdout writes
    class Tee:
        def __init__(self, stdout, log_file):
            self.stdout = stdout
            self.log_file = log_file

        def write(self, message):
            self.stdout.write(message)
            self.log_file.write(message)
            self.log_file.flush()

        def flush(self):
            self.stdout.flush()
            self.log_file.flush()

    def __del__(self):
        if hasattr(self, 'original_stdout') and hasattr(self, 'log_file'):
            sys.stdout = self.original_stdout
            self.log_file.close()
            print("inference complete, log file closed")

    def load_num_classes(self, csv_path):
        df = pd.read_csv(csv_path)
        return len(sorted(df['primary_label'].unique()))

    def get_segments(self, spec_tensor, seg_size=1000):
        """
        vectorizes the segmentation of the spectrogram using torch.unfold.
        assumes spec_tensor is of shape (1, F, T); returns a tensor of shape (num_segments, F, seg_size).
        """
        T = spec_tensor.shape[-1]
        remainder = T % seg_size
        if remainder:
            pad_size = seg_size - remainder
            spec_tensor = F.pad(spec_tensor, (0, pad_size))
        # after padding, use unfold along the time dimension
        # spec_tensor shape: (1, F, T_pad); unfolding gives shape (1, F, num_segments, seg_size)
        segments = spec_tensor.unfold(dimension=-1, size=seg_size, step=seg_size)
        # rearrange to shape (num_segments, F, seg_size)
        segments = segments.squeeze(0).permute(1, 0, 2)
        return segments

    def run_analysis(self):
        print("running inference with random predictions" if self.use_random_predictions else "running inference with onnx model")
        rows = []
        num_samples = len(self.val_wrapper.dataset)
        
        bird_classes = self.val_wrapper.unique_labels
        start_time = time.time()
        last_log_time = start_time
        samples_processed = 0
        segments_processed = 0

        # Process each file
        for i in tqdm(range(num_samples), desc="processing samples"):
            spec_tensor, label, label_idx, filenames = self.val_wrapper.next_batch()
            
            if label_idx == -1 or not filenames:
                continue
                
            # Get filename and extract base ID
            original_filename = filenames[0]
            base_id = re.sub(r'_segment_\d+.*$', '', original_filename)
            
            # Segment the spectrogram
            segments_batch = self.get_segments(spec_tensor, seg_size=1000)
            num_segments = segments_batch.size(0)
            
            # Generate predictions - use ONNX inference or random values
            if not self.use_random_predictions and self.ort_session is not None:
                # Real ONNX inference
                segment_probs = []
                for segment in segments_batch:
                    # Convert to numpy, add batch dimension (ONNX expects [batch, freq, time])
                    onnx_input = segment.numpy().astype(np.float32)
                    onnx_input = np.expand_dims(onnx_input, axis=0)  
                    
                    # Run inference
                    ort_inputs = {self.ort_session.get_inputs()[0].name: onnx_input}
                    ort_outputs = self.ort_session.run(None, ort_inputs)
                    
                    # Apply sigmoid to convert logits to probabilities
                    probs = 1.0 / (1.0 + np.exp(-ort_outputs[0]))
                    segment_probs.append(probs[0])  # Remove batch dimension
                    
                segment_probs = np.array(segment_probs)
            else:
                # Random predictions if no model is available
                segment_probs = np.random.rand(num_segments, len(bird_classes))
            
            # Generate a row for each segment
            for seg_idx in range(num_segments):
                time_marker = (seg_idx + 1) * 5  # each segment represents 5 sec
                row_id = f"{base_id}_{time_marker}"
                
                row_data = {"row_id": row_id}
                for cls_idx, class_name in enumerate(bird_classes):
                    row_data[class_name] = segment_probs[seg_idx, cls_idx]
                
                rows.append(row_data)
                segments_processed += 1

            samples_processed += 1
            current_time = time.time()
            if current_time - last_log_time > 10:
                elapsed = current_time - start_time
                samples_per_sec = samples_processed / elapsed
                segments_per_sec = segments_processed / elapsed
                print(f"speed stats: {samples_per_sec:.2f} samples/sec, {segments_per_sec:.2f} segments/sec, {elapsed:.2f}s total")
                print(f"current file: {base_id} with {num_segments} segments")
                last_log_time = current_time

        total_time = time.time() - start_time
        print("\nprocessing complete:")
        print(f"- total time: {total_time:.2f} seconds")
        print(f"- processed {samples_processed} samples")
        print(f"- generated {len(rows)} segment predictions")

        # Create DataFrame from rows 
        submission_df = pd.DataFrame(rows)
        
        # Check if we have any rows before proceeding
        if len(submission_df) == 0:
            print("WARNING: No rows generated for submission. Creating empty submission file.")
            submission_df = pd.DataFrame(columns=["row_id"] + list(bird_classes))
            submission_df.to_csv(self.args.submission_csv, index=False, float_format="%.4f")
            print(f"Empty csv file '{self.args.submission_csv}' created")
            return
        
        # Drop duplicate row_ids if any exist
        pre_dedup_count = len(submission_df)
        submission_df = submission_df.drop_duplicates(subset=["row_id"])
        post_dedup_count = len(submission_df)
        
        if pre_dedup_count != post_dedup_count:
            print(f"Dropped {pre_dedup_count - post_dedup_count} duplicate rows")
        
        # Filter out rows where time marker is outside the range 5-60
        def extract_time_marker(row_id):
            try:
                return int(row_id.split('_')[-1])
            except (ValueError, IndexError):
                return 0
        
        # Apply the filter - keep only rows with time markers between 5 and 60
        pre_filter_count = len(submission_df)
        valid_time_mask = submission_df['row_id'].apply(extract_time_marker).between(5, 60)
        submission_df = submission_df[valid_time_mask]
        post_filter_count = len(submission_df)
        
        if pre_filter_count != post_filter_count:
            print(f"Filtered out {pre_filter_count - post_filter_count} rows with time markers outside 5-60 range")
        
        print(f"- after filtering: {len(submission_df)} rows remain (removed duplicates and out-of-range time markers)")
        
        # Write to CSV with fixed float formatting
        submission_df.to_csv(self.args.submission_csv, index=False, float_format="%.4f")
        print(f"csv file '{self.args.submission_csv}' populated with model predictions")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "infer", "debug", "analyze"])
    parser.add_argument("--context_length", type=int, default=1000)
    
    # Training arguments
    parser.add_argument("--train_spec_dir", type=str, help="Directory containing training spectrograms")
    parser.add_argument("--val_spec_dir", type=str, help="Directory containing validation spectrograms")
    parser.add_argument("--taxonomy_file", type=str, help="Path to taxonomy file")
    parser.add_argument("--train_csv", type=str, help="Path to training CSV file")
    parser.add_argument("--output_dir", type=str, help="Directory to save model outputs", default=".")
    parser.add_argument("--pretrained_model_path", type=str, help="Path to pretrained model weights", default=None)
    parser.add_argument("--resume_checkpoint", type=str, help="Path to checkpoint for resuming training")
    parser.add_argument("--freeze_encoder", action="store_true", help="If set, only train the classifier and freeze the encoder weights. By default, both encoder and classifier are trained.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum training steps")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--save_interval", type=int, default=10, help="Steps between model checkpoints")
    parser.add_argument("--eval_interval", type=int, default=5, help="Steps between evaluations")
    parser.add_argument("--use_baseline", action="store_true", help="Whether to use baseline model instead of pretrained")
    parser.add_argument("--ema_alpha", type=float, default=0.2, help="EMA smoothing factor (0-1)")
    parser.add_argument("--save_best_metric", type=str, default="loss", choices=["loss", "roc_auc"], 
                       help="Metric to use for saving best model (loss or roc_auc)")
    parser.add_argument("--pool_type", type=str, default="mean", choices=["mean", "max"],
                       help="Type of pooling to use in the classifier")
    parser.add_argument("--submission_csv", type=str, default="submission.csv", 
                       help="Path to save the inference results CSV")
    parser.add_argument("--console_log", type=str, default="console.log",
                      help="Path to save the console output log file")
    parser.add_argument("--onnx_model_path", type=str, default=None,
                      help="Path to save the ONNX model file (defaults to output_dir/model.onnx)")
    
    
    args = parser.parse_args()

    if args.mode == "train":
        trainer = Trainer(args)
        trainer.train()
    elif args.mode == "analyze":
        analyzer = Analysis(args)
        analyzer.run_analysis()
    elif args.mode == "infer":
        infer = Inference(args)
        infer.run_analysis()

if __name__ == "__main__":
    main()
