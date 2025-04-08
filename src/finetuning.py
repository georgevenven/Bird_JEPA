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

# keeps track of label information <-> filename
class BirdCLEFDataWrapper:
    def __init__(self, csv_path, data_dir, context_length, batch_size):
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.context_length = context_length
        
        # Load CSV file into memory 
        self.csv_data = pd.read_csv(self.csv_path)
        
        # Extract primary_label and filename columns
        print(f"Processing CSV data from {csv_path}")
        self.primary_labels = self.csv_data['primary_label'].values
        self.filenames = self.csv_data['filename'].values if 'filename' in self.csv_data.columns else self.csv_data['file_name'].values
        
        # Get unique classes and create label mapping
        self.unique_labels = sorted(self.csv_data['primary_label'].unique())
        self.num_classes = len(self.unique_labels)
        self.label_to_idx = {label: i for i, label in enumerate(self.unique_labels)}
        
        print(f"Found {self.num_classes} unique bird classes")
        print(f"First 5 classes: {self.unique_labels[:5]}")
        
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
        self.init_dataset(context_length)

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

    def init_dataset(self, context_length):
        print(f"Initializing dataset from {self.data_dir}")
        self.dataset = BirdJEPA_Dataset(data_dir=self.data_dir, segment_len=context_length)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=data_class_collate_fn
        )
        self.dataloader_iter = iter(self.dataloader)
        print(f"Dataset initialized with {len(self.dataset)} samples")
    
    def next_batch(self):
        try:
            batch = next(self.dataloader_iter)
            
            # If we want to extract labels for the files in this batch
            if isinstance(batch, tuple) and len(batch) >= 6:
                filenames = batch[5]
                batch_labels = []
                batch_label_indices = []
                
                for filename in filenames:
                    label, label_idx = self.get_label_for_file(filename)
                    batch_labels.append(label)
                    batch_label_indices.append(label_idx)
                
                # Return batch along with extracted labels
                return batch, batch_labels, batch_label_indices
            
            # If no filenames to process, just return the batch with empty label lists
            return batch, [], []
        except StopIteration:
            # Reset iterator if we've gone through the entire dataset
            self.dataloader_iter = iter(self.dataloader)
            return self.next_batch()

class Classifier(nn.Module):
    def __init__(self, context_length, num_classes, hidden_dim=64):
        super(Classifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(context_length * hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.LogSoftmax(dim=1)
        )    

    def forward(self, x):
        return self.mlp(x)

class Model(nn.Module):
    def __init__(self, model_path=None, context_length=500, num_classes=1):
        super(Model, self).__init__()
        self.model_path = model_path
        self.context_length = context_length
        self.num_classes = num_classes
        self.encoder = None
        self.classifier = Classifier(context_length, num_classes, hidden_dim=64)
        
        # Load pretrained model if path is provided
        if model_path:
            self.encoder = load_model(model_path, load_weights=False)
        
        print(f"Model initialized with context_length={context_length}, num_classes={num_classes}")
    
    def forward(self, x):
        # Get embeddings from encoder
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        with torch.no_grad():  # Assuming we're freezing the encoder
            embedding_repr = self.encoder.inference_forward(x)

        embedding_repr = embedding_repr[0]
        outputs = self.classifier(embedding_repr.flatten(1,2))
        return outputs

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
            num_classes=self.train_wrapper.num_classes
        )
        
        # Set up optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.classifier.parameters(), 
            lr=self.args.learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.classifier.to(self.device)
        if self.model.encoder:
            self.model.encoder.to(self.device)
            
        # Initialize loss tracking
        self.train_losses = []
        self.val_losses = []
        self.train_losses_ema = []
        self.val_losses_ema = []
        self.eval_steps = []
        self.ema_alpha = args.ema_alpha if hasattr(args, 'ema_alpha') else 0.2  # EMA smoothing factor
        
        # Initialize ROC-AUC tracking
        self.val_roc_auc_scores = []
        self.val_roc_auc_scores_ema = []
        
        # Create output directory if it doesn't exist
        if args.output_dir:
            self.output_dir = Path(args.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Create a timestamp for this training run
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.output_dir / f"run_{self.timestamp}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
            self.run_dir = None
        
        # Get unique bird classes for ROC-AUC evaluation
        self.bird_classes = self.train_wrapper.unique_labels
    
    def calculate_ema(self, current_value, previous_ema=None):
        """Calculate Exponential Moving Average"""
        if previous_ema is None:
            return current_value
        return self.ema_alpha * current_value + (1 - self.ema_alpha) * previous_ema
    
    def train(self):
        # Training loop
        print(f"Starting training loop for {self.args.max_steps} steps")
        
        running_loss = 0.0
        best_val_loss = float('inf')
        best_roc_auc = 0.0
        patience_counter = 0
        
        for step in range(self.args.max_steps):
            # Get next batch
            batch_tuple, batch_labels, batch_label_indices = self.train_wrapper.next_batch()

            # Skip batches with missing labels
            if -1 in batch_label_indices:
                print(f"Skipping batch with missing labels")
                continue

            # Move data to device
            inputs = batch_tuple[0].to(self.device)
            labels = torch.tensor(batch_label_indices, dtype=torch.long).to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model.forward(inputs)
            
            # Calculate loss
            loss = self.criterion(outputs, labels)            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Print stats at evaluation interval
            if (step + 1) % self.args.eval_interval == 0:
                avg_train_loss = running_loss / self.args.eval_interval
                # Evaluate on validation set
                val_loss, roc_auc_score = self.evaluate(model=self.model, criterion=self.criterion, val_wrapper=self.val_wrapper)
                
                # Store losses
                self.train_losses.append(avg_train_loss)
                self.val_losses.append(val_loss)
                self.val_roc_auc_scores.append(roc_auc_score)
                self.eval_steps.append(step + 1)
                
                # Calculate EMA
                if len(self.train_losses_ema) == 0:
                    self.train_losses_ema.append(avg_train_loss)
                    self.val_losses_ema.append(val_loss)
                    self.val_roc_auc_scores_ema.append(roc_auc_score)
                else:
                    self.train_losses_ema.append(self.calculate_ema(avg_train_loss, self.train_losses_ema[-1]))
                    self.val_losses_ema.append(self.calculate_ema(val_loss, self.val_losses_ema[-1]))
                    self.val_roc_auc_scores_ema.append(self.calculate_ema(roc_auc_score, self.val_roc_auc_scores_ema[-1]))
                
                print(f"Step {step+1}/{self.args.max_steps}, Training Loss: {avg_train_loss:.4f}, "
                      f"Validation Loss: {val_loss:.4f}, ROC-AUC: {roc_auc_score:.4f}")
                
                # Save model if it has the best validation score
                if self.args.save_best_metric == 'loss' and val_loss < best_val_loss and self.run_dir:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model(step, "best_loss")
                elif self.args.save_best_metric == 'roc_auc' and roc_auc_score > best_roc_auc and self.run_dir:
                    best_roc_auc = roc_auc_score
                    patience_counter = 0
                    self.save_model(step, "best_roc_auc")
                else:
                    patience_counter += 1
                
                # Save model periodically
                if (step + 1) % self.args.save_interval == 0 and self.run_dir:
                    self.save_model(step, "checkpoint")
                
                # Early stopping
                if patience_counter >= self.args.early_stopping_patience:
                    print(f"Early stopping triggered at step {step+1}")
                    break
                
                # Reset running loss
                running_loss = 0.0
        
        # After training is complete, plot and save loss curves
        self.save_loss_data()
        self.plot_losses()
        self.plot_roc_auc()
        
        # Return to eval mode
        self.model.eval()
    
    def save_model(self, step, prefix="checkpoint"):
        """Save model checkpoint"""
        if not self.run_dir:
            return
            
        save_path = self.run_dir / f"{prefix}_step_{step+1}.pt"
        torch.save({
            'step': step + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'val_loss': self.val_losses[-1] if self.val_losses else None,
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def save_loss_data(self):
        """Save loss data to JSON file"""
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
        
        json_path = self.run_dir / 'metrics_data.json'
        with open(json_path, 'w') as f:
            json.dump(loss_data, f, indent=4)
        
        # Also save as CSV for easier analysis
        csv_path = self.run_dir / 'metrics_data.csv'
        df = pd.DataFrame({
            'step': self.eval_steps,
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_loss_ema': self.train_losses_ema,
            'val_loss_ema': self.val_losses_ema,
            'val_roc_auc': self.val_roc_auc_scores,
            'val_roc_auc_ema': self.val_roc_auc_scores_ema
        })
        df.to_csv(csv_path, index=False)
        
        print(f"Metrics data saved to {json_path} and {csv_path}")
    
    def plot_losses(self):
        """Plot and save the loss curves"""
        if not self.run_dir:
            return
            
        plt.figure(figsize=(10, 6))
        
        # Plot raw losses
        plt.plot(self.eval_steps, self.train_losses, 'b-', alpha=0.3, label='Train Loss')
        plt.plot(self.eval_steps, self.val_losses, 'r-', alpha=0.3, label='Validation Loss')
        
        # Plot smoothed losses
        plt.plot(self.eval_steps, self.train_losses_ema, 'b-', linewidth=2, label='Train Loss (EMA)')
        plt.plot(self.eval_steps, self.val_losses_ema, 'r-', linewidth=2, label='Validation Loss (EMA)')
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plot_path = self.run_dir / 'loss_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Loss plot saved to {plot_path}")
    
    def plot_roc_auc(self):
        """Plot and save the ROC-AUC curves"""
        if not self.run_dir or not self.val_roc_auc_scores:
            return
            
        plt.figure(figsize=(10, 6))
        
        # Plot raw ROC-AUC
        plt.plot(self.eval_steps, self.val_roc_auc_scores, 'g-', alpha=0.3, label='Validation ROC-AUC')
        
        # Plot smoothed ROC-AUC
        plt.plot(self.eval_steps, self.val_roc_auc_scores_ema, 'g-', linewidth=2, label='Validation ROC-AUC (EMA)')
        
        plt.title('Validation ROC-AUC Score')
        plt.xlabel('Steps')
        plt.ylabel('ROC-AUC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set y-axis limits for ROC-AUC (typically between 0.5 and 1.0)
        plt.ylim(0.5, 1.05)
        
        # Save the plot
        plot_path = self.run_dir / 'roc_auc_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC-AUC plot saved to {plot_path}")

    def evaluate(self, model, criterion, val_wrapper):
        """Evaluate model on validation set, returning both loss and ROC-AUC score"""
        model.eval()
        total_loss = 0.0
        num_batches = 5  # Evaluate on multiple batches for better validation statistics
        
        all_labels = []  # True labels
        all_preds = []   # Predicted probabilities
        
        with torch.no_grad():
            for _ in range(num_batches):
                batch_tuple, batch_labels, batch_label_indices = val_wrapper.next_batch()
                
                # Skip batches with missing labels
                if -1 in batch_label_indices:
                    continue
                    
                # Move data to device
                inputs = batch_tuple[0].to(self.device)
                labels = torch.tensor(batch_label_indices, dtype=torch.long).to(self.device)
                
                # Forward pass
                outputs = model.forward(inputs)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                # Convert to probabilities
                if outputs.shape[1] > 1:  # Multi-class case
                    # If using LogSoftmax output, convert to actual probabilities
                    if torch.min(outputs) < 0:
                        probs = torch.exp(outputs)
                    else:
                        probs = outputs
                    
                    # Convert one-hot indices to one-hot encoded format for ROC-AUC calculation
                    batch_size = labels.size(0)
                    num_classes = probs.size(1)
                    
                    # Create one-hot encoded labels
                    labels_one_hot = torch.zeros(batch_size, num_classes).to(self.device)
                    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
                    
                    all_labels.append(labels_one_hot.cpu().numpy())
                    all_preds.append(probs.cpu().numpy())
                else:  # Binary case
                    all_labels.append(labels.cpu().numpy())
                    all_preds.append(outputs.cpu().numpy())
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        
        # Calculate ROC-AUC if we have predictions
        roc_auc = 0.0
        if all_labels and all_preds:
            try:
                all_labels = np.vstack(all_labels)
                all_preds = np.vstack(all_preds)
                
                # Create DataFrame format for Kaggle scoring function
                if all_labels.shape[1] > 1:  # Multi-class case
                    solution_df = pd.DataFrame(all_labels, columns=self.bird_classes)
                    submission_df = pd.DataFrame(all_preds, columns=self.bird_classes)
                    
                    # Add row_id column for Kaggle scoring format
                    row_ids = [f"row_{i}" for i in range(len(solution_df))]
                    solution_df.insert(0, "row_id", row_ids)
                    submission_df.insert(0, "row_id", row_ids)
                    
                    # Calculate ROC-AUC using Kaggle's function
                    roc_auc = kaggle_score(solution_df.copy(), submission_df.copy(), "row_id")
                else:  # Binary case
                    roc_auc = safe_roc_auc_score(all_labels, all_preds)
            except Exception as e:
                print(f"Error calculating ROC-AUC: {e}")
        
        model.train()  # Set model back to training mode
        return avg_loss, roc_auc

class Analysis:
    pass

class Inference:
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "infer", "debug", "analyze"])
    parser.add_argument("--context_length", type=int, default=500)
    
    # Training arguments
    parser.add_argument("--train_spec_dir", type=str, help="Directory containing training spectrograms")
    parser.add_argument("--val_spec_dir", type=str, help="Directory containing validation spectrograms")
    parser.add_argument("--taxonomy_file", type=str, help="Path to taxonomy file")
    parser.add_argument("--train_csv", type=str, help="Path to training CSV file")
    parser.add_argument("--val_csv", type=str, help="Path to validation CSV file")  # Added val_csv parameter
    parser.add_argument("--output_dir", type=str, help="Directory to save model outputs")
    parser.add_argument("--pretrained_model_path", type=str, help="Path to pretrained model weights")
    parser.add_argument("--freeze_encoder", action="store_true", help="Whether to freeze encoder weights")
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

    args = parser.parse_args()

    if args.mode == "train":
        trainer = Trainer(args)
        trainer.train()

if __name__ == "__main__":
    main()
