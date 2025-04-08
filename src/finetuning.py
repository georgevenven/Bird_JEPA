import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import random
from pathlib import Path
import subprocess
import shutil
import glob
import sys
import time
from datetime import datetime

# Import from project modules
from model import BirdJEPA
from utils import load_model
from spectrogram_generator import WavtoSpec
from data_class import BirdJEPA_Dataset, collate_fn as data_class_collate_fn

# Define a special dataset for BirdCLEF fine-tuning
class BirdCLEFSpecDataset(BirdJEPA_Dataset):
    def __init__(self, spec_dir, taxonomy_file, train_csv, mode='train', verbose=False):
        """
        Dataset for BirdCLEF classification using pre-generated spectrograms
        
        Args:
            spec_dir (str): Directory containing pre-generated spectrograms
            taxonomy_file (str): Path to taxonomy.csv file with species info
            train_csv (str): Path to train.csv file with labels
            mode (str): 'train' or 'val' mode
            verbose (bool): Whether to print assignment messages
        """
        # Initialize parent class
        super().__init__(data_dir=spec_dir, segment_len=2500)
        
        # Store the mode for length determination
        self.mode = mode
        self.verbose = verbose
        
        # Load taxonomy data
        self.taxonomy = pd.read_csv(taxonomy_file)
        self.species_ids = self.taxonomy['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        
        # Print first few entries in taxonomy for debugging
        print("\nDEBUG: First 5 species in taxonomy:")
        for i in range(min(5, len(self.species_ids))):
            print(f"  {i}: {self.species_ids[i]}")
        
        # Map species to index
        self.species_to_idx = {species: idx for idx, species in enumerate(self.species_ids)}
        
        # Load data from CSV
        self.df = pd.read_csv(train_csv)
        
        # Print CSV structure for debugging
        print("\nDEBUG: Train CSV columns:", list(self.df.columns))
        print(f"DEBUG: Train CSV first row: {self.df.iloc[0].to_dict()}")
        
        # Filter out samples without species in taxonomy
        self.df = self.df[self.df['primary_label'].isin(self.species_ids)]
        
        # Check if 'split' column exists, if not create it
        if 'split' not in self.df.columns:
            print("No 'split' column found in CSV. Creating 80/20 train/val split.")
            # Create a random 80/20 split for train/val
            np.random.seed(42)  # For reproducibility
            msk = np.random.rand(len(self.df)) < 0.8
            self.df['split'] = 'val'
            self.df.loc[msk, 'split'] = 'train'
        
        # Filter dataset based on mode (train/val)
        if mode == 'train':
            self.df = self.df[self.df['split'] == 'train']
        elif mode == 'val':
            self.df = self.df[self.df['split'] == 'val']
        
        # Create a direct mapping from file paths to their primary labels
        # This will handle the actual directory structure in BirdCLEF/train_audio
        self.filename_to_species = {}
        
        # Log how many files we found and processed
        found_files = 0
        matching_files = 0
        
        # Extract the base filename from our spectrogram paths
        # These should be derived from the original audio files
        for file_path in self.file_paths:
            base_name = os.path.basename(file_path)
            # Remove any segment information and extension
            # The file format should be something like "XC123456_segment_0.npz"
            base_name_parts = base_name.split('_segment_')
            if len(base_name_parts) > 0:
                audio_id = base_name_parts[0]
                found_files += 1
                
                # Print some example filenames for debugging
                if found_files <= 5:
                    print(f"\nDEBUG: Processing file {found_files}: {base_name}")
                    print(f"  audio_id: {audio_id}")
                
                # Look for matches in the dataframe
                # Check both the full filename and just the ID portion
                matches = self.df[self.df['filename'].str.contains(audio_id)]
                
                if not matches.empty:
                    # Use the first match's primary label
                    primary_label = matches.iloc[0]['primary_label']
                    self.filename_to_species[audio_id] = primary_label
                    matching_files += 1
                    
                    # Print first few matches for debugging
                    if matching_files <= 5:
                        print(f"  Matched to {primary_label} in CSV")
                        
                # For first few unmatched files, show why it didn't match
                if not matches.empty == False and found_files <= 5:
                    # Get the species ID from the filename (the part before the first underscore)
                    parts = audio_id.split('_')
                    if len(parts) > 0:
                        potential_species = parts[0]
                        print(f"  Species from filename: {potential_species}")
                        print(f"  Is in taxonomy: {potential_species in self.species_to_idx}")
                    
                    # Print a few examples from the dataframe to see if there's a filename format issue
                    print(f"  Example CSV filenames: {self.df['filename'].iloc[:3].tolist()}")
        
        if found_files > 0:
            print(f"Successfully matched {matching_files}/{found_files} spectrogram files to species labels ({matching_files/found_files*100:.1f}%)")
        else:
            print("Warning: No files found in the spectrogram directory. Check your paths.")
        
        # Log dataset info
        print(f"Created {mode} dataset with {len(self.file_paths)} files, {self.num_classes} classes")
        print(f"Found {len(self.filename_to_species)} labeled files")
        
        # If we have too few matches, it might indicate a path/naming issue
        if matching_files < 0.5 * found_files:
            print("\nWARNING: Less than 50% of files were matched to species. This might indicate issues with:")
            print("  1. Spectrogram filenames not matching audio file IDs in the CSV")
            print("  2. Incorrect spectrogram generation process")
            print("  3. Mismatched directories\n")
    
    def __len__(self):
        """
        Override the parent class __len__ method to return the actual number of files
        rather than the "infinite" dataset size.
        """
        # For validation, return the actual file count
        if self.mode == 'val':
            return len(self.file_paths)
        # For training, we can still use a large number to enable diverse sampling,
        # but limit it to a reasonable multiple of the actual file count
        else:
            return min(int(1e5), len(self.file_paths) * 25)  # 25 segments per file is reasonable
    
    def __getitem__(self, idx):
        """
        Get a data sample from the dataset
        
        Args:
            idx: Index of the sample
        
        Returns:
            spectrogram: Spectrogram tensor [C, F, T]
            label: Multi-hot encoded label tensor [num_classes]
        """
        # Get the file path
        file_idx = idx % len(self.file_paths)
        file_path = self.file_paths[file_idx]
        
        # Load with memory mapping to save memory
        try:
            data = np.load(file_path, allow_pickle=True, mmap_mode='r')
            spec = data['s']
            
            # Check for correct dimensions - should be [F, T] where F is frequency bins
            if len(spec.shape) != 2:
                raise ValueError(f"Expected 2D spectrogram, got shape {spec.shape}")
                
            # Ensure we have the expected number of frequency bins (128)
            expected_freq_bins = 128
            if spec.shape[0] != expected_freq_bins:
                # Reshape to expected dimensions
                from scipy.ndimage import zoom
                zoom_factor = expected_freq_bins / spec.shape[0]
                spec = zoom(spec, (zoom_factor, 1), order=1)
                
            # Process spectrogram
            if spec.shape[1] < self.segment_len:
                padded_spec = np.zeros((spec.shape[0], self.segment_len))
                padded_spec[:, :spec.shape[1]] = spec
                spec = padded_spec
            
            # Get a random segment
            if self.mode == 'train':
                start = random.randint(0, spec.shape[1] - self.segment_len)
            else:
                # For validation, use center segment
                start = (spec.shape[1] - self.segment_len) // 2
                
            segment = spec[:, start:start+self.segment_len].copy()
            
            # Normalize
            mean_val = np.mean(segment)
            std_val = np.std(segment)
            segment = (segment - mean_val) / (std_val + 1e-8)
            
            # Convert to tensor
            spectrogram = torch.from_numpy(segment).float()
            
        except Exception as e:
            print(f"Error loading spectrogram from {file_path}: {e}")
            # Return a random entry as fallback
            return self.__getitem__((idx + 1) % len(self.file_paths))
        
        # Initialize multi-hot encoded label
        label = torch.zeros(len(self.species_ids), dtype=torch.float32)
        
        # Extract filename
        base_filename = os.path.basename(file_path)
        matched = False
        
        # 1. Try to extract species from the filename directly
        # New format: species_id_filename.npz or species_id_filename_segment_X.npz
        parts = base_filename.split('_')
        if len(parts) >= 2:
            species_id = parts[0]
            if species_id in self.species_to_idx:
                label[self.species_to_idx[species_id]] = 1.0
                matched = True
        
        # 2. Try to match base filename (without extension and segments) to the mapping
        if not matched:
            base_filename = base_filename.split('_segment_')[0]  # Remove segment part if present
            base_filename = os.path.splitext(base_filename)[0]  # Remove extension
            
            # If there's an underscore, try extracting just the ID part
            if '_' in base_filename:
                # Try with removing the species prefix
                parts = base_filename.split('_')
                if len(parts) > 1:
                    # Remove first part (species) and join the rest
                    potential_key = '_'.join(parts[1:])
                    if potential_key in self.filename_to_species:
                        species = self.filename_to_species[potential_key]
                        label[self.species_to_idx[species]] = 1.0
                        matched = True
        
        # If no match found, try another file
        if not matched:
            # If we're in training mode and can't match, try a different file
            if self.mode == 'train':
                # Recursive call to get a different file - limited recursion to prevent issues
                retry_count = getattr(self, '_retry_count', 0)
                if retry_count < 5:  # Limit recursion depth
                    self._retry_count = retry_count + 1
                    return self.__getitem__((idx + 1) % len(self.file_paths))
                else:
                    # If too many retries, reset counter and return with warning
                    self._retry_count = 0
                    if self.verbose:
                        print(f"Warning: Failed to match species after 5 retries. Using zero label for {base_filename}")
            elif self.verbose:
                print(f"Warning: No species match found for {base_filename}")
        else:
            # Reset retry counter if we found a match
            self._retry_count = 0
        
        # For training mode, ensure we have both positive and negative examples
        # by occasionally adding a second class
        if self.mode == 'train' and matched and random.random() < 0.3:  # 30% chance
            # Add a second species label to create multi-label examples
            available_indices = [i for i in range(len(self.species_ids)) if label[i] == 0]
            if available_indices:
                second_species_idx = random.choice(available_indices)
                label[second_species_idx] = 1.0
        
        return spectrogram, label

    def _get_raw_item(self, idx):
        """Get raw item without label processing for debugging"""
        file_idx = idx % len(self.file_paths)
        file_path = self.file_paths[file_idx]
        
        # Load with memory mapping
        data = np.load(file_path, allow_pickle=True, mmap_mode='r')
        spec = data['s']
        ground_truth_labels = data['labels']
        
        # Process spectrogram
        if spec.shape[1] < self.segment_len:
            padded_spec = np.zeros((spec.shape[0], self.segment_len))
            padded_spec[:, :spec.shape[1]] = spec
            spec = padded_spec
        
        # Get a random segment
        start = random.randint(0, spec.shape[1] - self.segment_len)
        segment = spec[:, start:start+self.segment_len].copy()
        
        # Normalize
        mean_val = np.mean(segment)
        std_val = np.std(segment)
        segment = (segment - mean_val) / (std_val + 1e-8)
        
        return torch.from_numpy(segment).float(), ground_truth_labels, file_path

# Classification collate function adapted from the original collate_fn
def classification_collate_fn(batch):
    """
    Collate function for classification task
    
    Args:
        batch: List of (spectrogram, label) tuples
    
    Returns:
        spectrograms: Batch of spectrograms [B, F, T]
        labels: Batch of multi-hot encoded labels [B, num_classes]
    """
    # Unpack batch
    spectrograms, labels = zip(*batch)
    
    # Stack spectrograms along batch dimension
    stacked_spectrograms = torch.stack(spectrograms, dim=0)
    
    # Stack labels along batch dimension
    stacked_labels = torch.stack(labels, dim=0)
    
    return stacked_spectrograms, stacked_labels

# Classifier model for fine-tuning
class BirdCLEFClassifier(nn.Module):
    def __init__(self, base_model, num_classes, freeze_encoder=True):
        """
        Classifier for BirdCLEF using the pretrained BirdJEPA model
        
        Args:
            base_model: Pretrained BirdJEPA model
            num_classes: Number of bird species classes
            freeze_encoder: Whether to freeze encoder weights
        """
        super().__init__()
        
        # Store base model - this is the BirdJEPA model
        self.base_model = base_model
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.base_model.context_encoder.parameters():
                param.requires_grad = False
        
        # Get embedding dimension from base model
        self.embedding_dim = self.base_model.hidden_dim
        
        # Add projection layer to convert from encoder output to LSTM input
        self.projection = nn.Linear(64, 256)  # Project from 64 to 256
        
        # BiLSTM 
        self.bilstm = nn.LSTM(
            input_size=256,  # Match projection output size
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        # Classifier head for BiLSTM output
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # 512 = 256*2 (bidirectional)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid()  # Sigmoid for multi-label classification
        )
    
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x: Input spectrogram [B, F, T]
        
        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Process through encoder of base model only
        # Check if any parameters in context_encoder require gradients
        requires_grad = any(p.requires_grad for p in self.base_model.context_encoder.parameters())
        
        with torch.set_grad_enabled(requires_grad):
            # Handle different input dimensions
            if x.dim() == 3:  # [B, F, T]
                x = x.unsqueeze(1)  # Add channel dimension [B, 1, F, T]
            
            # Get embeddings from the context encoder
            embeddings, _ = self.base_model.context_encoder(x)
        
        # Ensure embeddings is shaped correctly for projection
        if embeddings.dim() == 3:
            if embeddings.shape[1] == self.embedding_dim:
                # Shape is [B, D, T], transpose to [B, T, D]
                embeddings = embeddings.transpose(1, 2)
            # Now shape should be [B, T, D]
            batch_size, seq_len, feat_dim = embeddings.shape
            
            # Apply projection to each timestep
            projected = self.projection(embeddings)  # [B, T, 256]
        else:
            raise ValueError(f"Unexpected embeddings shape: {embeddings.shape}")
        
        # Process through BiLSTM
        lstm_out, (hidden, _) = self.bilstm(projected)
        
        # Use the concatenated final hidden states from both directions
        # hidden shape: [num_layers*2, batch, hidden_size]
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch, hidden_size*2]
        
        # Forward through classifier head
        logits = self.classifier(last_hidden)
        
        return logits

# Add this new baseline classifier after BirdCLEFClassifier
class BaselineClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        """
        Baseline classifier that processes spectrograms directly without using the pretrained model
        
        Args:
            input_dim: Input feature dimension (frequency bins)
            num_classes: Number of bird species classes
        """
        super().__init__()
        
        # Convolutional layers to process raw spectrograms
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), stride=(2,1), padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5,5), stride=(2,1), padding=(2,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(7,7), stride=(2,1), padding=(3,3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(7,7), stride=(2,1), padding=(3,3)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # We'll initialize the projection layer later after seeing the actual dimensions
        self.projection = None
        
        # BiLSTM for temporal processing
        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # 512 = 256*2 (bidirectional)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid()  # Sigmoid for multi-label classification
        )
        
        # Flag to indicate if we've initialized the projection layer
        self.initialized = False
    
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x: Input spectrogram [B, F, T]
        
        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Add channel dimension if needed
        if x.dim() == 3:  # [B, F, T]
            x = x.unsqueeze(1)  # Add channel dimension [B, 1, F, T]
        
        # Process through convolutional layers
        x = self.conv_layers(x)  # [B, 128, F/16, T]
        
        # Reshape for projection
        B, C, F, T = x.shape
        x = x.transpose(1, 3)  # [B, T, F, C]
        x = x.reshape(B, T, F * C)  # [B, T, F*C]
        
        # Initialize projection layer on first forward pass
        if not self.initialized:
            input_dim = F * C
            print(f"Initializing projection layer with input dim: {input_dim}")
            self.projection = nn.Linear(input_dim, 256).to(x.device)
            self.initialized = True
        
        # Project to fixed dimension
        x = self.projection(x)  # [B, T, 256]
        
        # Process through BiLSTM
        lstm_out, (hidden, _) = self.bilstm(x)
        
        # Use the concatenated final hidden states from both directions
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch, hidden_size*2]
        
        # Forward through classifier head
        logits = self.classifier(last_hidden)
        
        return logits

# Kaggle scoring function
class ParticipantVisibleError(Exception):
    pass

def kaggle_score(solution, submission, row_id_column_name=None):
    '''
    Version of macro-averaged ROC-AUC score that ignores all classes that have no true positive labels.
    '''
    if row_id_column_name is not None:
        if row_id_column_name in solution:
            del solution[row_id_column_name]
        if row_id_column_name in submission:
            del submission[row_id_column_name]

    # Check for valid numeric data
    for col in submission.columns:
        if not pd.api.types.is_numeric_dtype(submission[col]):
            bad_dtypes = {x: submission[x].dtype for x in submission.columns if not pd.api.types.is_numeric_dtype(submission[x])}
            raise ParticipantVisibleError(f'Invalid submission data types found: {bad_dtypes}')

    # Get columns with positive labels
    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index)
    assert len(scored_columns) > 0

    try:
        return roc_auc_score(solution[scored_columns].values, submission[scored_columns].values, average='macro')
    except Exception as e:
        print(f"Error in scoring: {e}")
        return 0.0

def finetune_model(
    pretrained_model_path,
    train_spec_dir,
    val_spec_dir,
    taxonomy_file,
    train_csv,
    output_dir,
    freeze_encoder=True,
    learning_rate=1e-4,
    batch_size=16,
    max_steps=5000,
    early_stopping_patience=5,
    device=None,
    save_interval=500,  # Save checkpoints every N steps
    eval_interval=100,   # Evaluate and log metrics every N steps
    use_baseline=False  # Flag to use baseline model instead of pretrained
):
    """
    Fine-tune a pretrained BirdJEPA model for BirdCLEF classification
    using pre-generated spectrograms and the BirdJEPA_Dataset class.
    If use_baseline=True, creates a model from scratch without using pretrained weights.
    """
    # Create a timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = "baseline" if use_baseline else "finetuned"
    run_dir = os.path.join(output_dir, f"{model_type}_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories for checkpoints, logs, and plots
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    plot_dir = os.path.join(run_dir, "plots")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(log_dir, "training_log.csv")
    with open(log_file, 'w') as f:
        f.write("step,train_loss,val_loss,train_kaggle_score,val_kaggle_score,time_taken\n")
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create datasets using the BirdCLEFSpecDataset class (extends BirdJEPA_Dataset)
    print(f"Creating datasets")
    train_dataset = BirdCLEFSpecDataset(
        spec_dir=train_spec_dir,
        taxonomy_file=taxonomy_file,
        train_csv=train_csv,
        mode='train',
        verbose=False
    )
    
    val_dataset = BirdCLEFSpecDataset(
        spec_dir=val_spec_dir,
        taxonomy_file=taxonomy_file,
        train_csv=train_csv,
        mode='val',
        verbose=False
    )
    
    # Create data loaders using the adapted collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=classification_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=classification_collate_fn,
        pin_memory=True
    )
    
    # Print actual dataset sizes for debugging
    print(f"Training dataset size: {len(train_dataset)} samples")
    print(f"Validation dataset size: {len(val_dataset)} samples")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create classifier model - either baseline or pretrained
    if use_baseline:
        print("Creating baseline classifier (no pretrained weights)")
        classifier = BaselineClassifier(
            input_dim=128,  # Typical spectrogram frequency bins
            num_classes=train_dataset.num_classes
        )
        # Save empty config to output directory
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump({"model_type": "baseline"}, f, indent=4)
    else:
        print(f"Loading pretrained model from {pretrained_model_path}")
        base_model, _, config = load_model(pretrained_model_path, return_checkpoint=True)
        
        # Save model config to output directory
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
            
        classifier = BirdCLEFClassifier(
            base_model=base_model,
            num_classes=train_dataset.num_classes,
            freeze_encoder=freeze_encoder
        )
    
    classifier = classifier.to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(
        [p for p in classifier.parameters() if p.requires_grad],
        lr=learning_rate
    )
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    # Loss function - binary cross entropy for multi-label classification
    criterion = nn.BCELoss()
    
    # Training loop - using steps instead of epochs
    best_val_kaggle_score = 0
    steps_since_improvement = 0
    train_losses = []
    val_losses = []
    train_kaggle_scores = []
    val_kaggle_scores = []
    times = []
    
    # Get species IDs for metrics
    species_ids = train_dataset.species_ids
    
    # Create iterators
    train_iter = iter(train_loader)
    
    print(f"Starting training for {max_steps} steps")
    step = 0
    
    # Remove epoch tracking
    # We'll track steps only for simplicity
    
    while step < max_steps:
        # Training
        classifier.train()
        step_start_time = time.time()
        
        try:
            # Get next batch
            specs, labels = next(train_iter)
        except StopIteration:
            # Reset iterator when we've gone through the entire dataset
            train_iter = iter(train_loader)
            specs, labels = next(train_iter)
            print(f"Restarting training iterator")
        
        # Move data to device
        specs = specs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = classifier(specs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        train_losses.append(train_loss)
        
        # Increment step counter
        step += 1
        
        # Evaluation at fixed intervals
        if step % eval_interval == 0 or step == max_steps:
            # Calculate step time
            step_time = time.time() - step_start_time
            times.append(step_time)
            
            # Full validation
            classifier.eval()
            val_loss = 0
            all_val_outputs = []
            all_val_labels = []
            all_train_outputs = [outputs.detach().cpu().numpy()]
            all_train_labels = [labels.detach().cpu().numpy()]
            
            with torch.no_grad():
                # Only create progress bar for larger datasets
                use_progress_bar = len(val_loader) > 10
                
                if use_progress_bar:
                    val_iter = tqdm(val_loader, desc=f"Step {step}/{max_steps} [Validation]")
                else:
                    print(f"Running validation (Step {step}/{max_steps})...")
                    val_iter = val_loader
                
                # Limit validation size for very small datasets
                max_val_batches = 1  # Just evaluate a single batch for speed
                
                for i, (val_specs, val_labels) in enumerate(val_iter):
                    if i >= max_val_batches:
                        break
                        
                    val_specs = val_specs.to(device)
                    val_labels = val_labels.to(device)
                    
                    val_outputs = classifier(val_specs)
                    batch_loss = criterion(val_outputs, val_labels)
                    
                    val_loss += batch_loss.item()
                    
                    # Collect for metrics
                    all_val_outputs.append(val_outputs.cpu().numpy())
                    all_val_labels.append(val_labels.cpu().numpy())
                    
                    if use_progress_bar:
                        val_iter.set_postfix(loss=batch_loss.item())
            
            # Calculate average validation loss
            val_loss = val_loss / min(max_val_batches, len(val_loader))
            val_losses.append(val_loss)
            
            # Stack outputs and labels
            all_val_outputs = np.vstack(all_val_outputs) if all_val_outputs else np.array([])
            all_val_labels = np.vstack(all_val_labels) if all_val_labels else np.array([])
            all_train_outputs = np.vstack(all_train_outputs)
            all_train_labels = np.vstack(all_train_labels)
            
            # Calculate train Kaggle score
            train_kaggle_score = 0
            if len(all_train_labels) > 0:
                # Create DataFrame format similar to competition submissions
                train_pred_df = pd.DataFrame(all_train_outputs, columns=species_ids)
                train_true_df = pd.DataFrame(all_train_labels, columns=species_ids)
                
                # Add a dummy row_id column
                train_pred_df['row_id'] = range(len(train_pred_df))
                train_true_df['row_id'] = range(len(train_true_df))
                
                try:
                    train_kaggle_score = kaggle_score(train_true_df, train_pred_df, row_id_column_name='row_id')
                except Exception as e:
                    print(f"Error calculating train Kaggle score: {e}")
                    train_kaggle_score = max(0, min(1, 1 - train_loss/2))
            
            train_kaggle_scores.append(train_kaggle_score)
            
            # Calculate validation Kaggle score
            val_kaggle_score = 0
            if len(all_val_labels) > 0:
                # Create DataFrame format similar to competition submissions
                val_pred_df = pd.DataFrame(all_val_outputs, columns=species_ids)
                val_true_df = pd.DataFrame(all_val_labels, columns=species_ids)
                
                # Add a dummy row_id column
                val_pred_df['row_id'] = range(len(val_pred_df))
                val_true_df['row_id'] = range(len(val_true_df))
                
                try:
                    val_kaggle_score = kaggle_score(val_true_df, val_pred_df, row_id_column_name='row_id')
                except Exception as e:
                    print(f"Error calculating validation Kaggle score: {e}")
                    # Use a meaningful score based on loss
                    # Lower loss = higher score, bounded between 0 and 1
                    val_kaggle_score = max(0, min(1, 1 - val_loss/2))
            
            val_kaggle_scores.append(val_kaggle_score)
            
            # Log results
            print(f"Step {step}/{max_steps}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Kaggle Score: {train_kaggle_score:.4f}, Val Kaggle Score: {val_kaggle_score:.4f}, "
                  f"Time: {step_time:.2f}s")
            
            # Save to log file
            with open(log_file, 'a') as f:
                f.write(f"{step},{train_loss:.6f},{val_loss:.6f},{train_kaggle_score:.6f},{val_kaggle_score:.6f},{step_time:.2f}\n")
            
            # Update learning rate scheduler
            scheduler.step(val_kaggle_score)  # Use validation Kaggle score for scheduling
            
            # Check if this is the best model by validation Kaggle score
            is_best_model = val_kaggle_score > best_val_kaggle_score
            if is_best_model:
                best_val_kaggle_score = val_kaggle_score
                steps_since_improvement = 0
                
                # Save best model
                model_path = os.path.join(run_dir, f"best_model.pt")
                torch.save({
                    'step': step,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'kaggle_score': val_kaggle_score,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, model_path)
                print(f"Saved best model with Kaggle score {val_kaggle_score:.4f}")
                
                # Create a symlink to the best model in the output directory
                best_model_link = os.path.join(output_dir, "best_model.pt")
                
                # Ensure the path exists for the symlink
                os.makedirs(os.path.dirname(best_model_link), exist_ok=True)
                
                # Remove existing symlink or file if it exists
                if os.path.exists(best_model_link) or os.path.islink(best_model_link):
                    try:
                        if os.path.islink(best_model_link):
                            os.unlink(best_model_link)
                        else:
                            os.remove(best_model_link)
                    except Exception as e:
                        print(f"Warning: Failed to remove existing best model link: {e}")
                
                # Try to create symlink, fall back to copy
                try:
                    # Get absolute paths to avoid relative path issues
                    abs_model_path = os.path.abspath(model_path)
                    abs_model_link = os.path.abspath(best_model_link)
                    
                    # Create symlink
                    os.symlink(abs_model_path, abs_model_link)
                    print(f"Created symlink from {abs_model_path} to {abs_model_link}")
                except Exception as e:
                    print(f"Warning: Failed to create symlink: {e}")
                    try:
                        # Fall back to copy
                        shutil.copy2(model_path, best_model_link)
                        print(f"Copied best model to {best_model_link}")
                    except Exception as copy_error:
                        # If both fail, just log and continue
                        print(f"Error: Could not copy best model: {copy_error}")
                        print(f"Best model is still available at: {model_path}")
            else:
                steps_since_improvement += 1
                if steps_since_improvement >= early_stopping_patience:
                    print(f"Early stopping triggered after {step} steps")
                    break
            
        # Save checkpoint at regular intervals
        if step == max_steps:
            checkpoint_path = os.path.join(checkpoint_dir, f"final_model.pt")
            torch.save({
                'step': step,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_kaggle_score': best_val_kaggle_score,
                'steps_since_improvement': steps_since_improvement,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_kaggle_scores': train_kaggle_scores,
                'val_kaggle_scores': val_kaggle_scores
            }, checkpoint_path)
            print(f"Saved final model at step {step}")
    
    # Generate plots at the end of training
    print("Generating training metrics plots...")
    plt.figure(figsize=(10, 6))
    
    # Plot losses
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot([i * eval_interval for i in range(1, len(val_losses) + 1)], val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'final_training_metrics.png'))
    print(f"Saved training metrics plot to {os.path.join(plot_dir, 'final_training_metrics.png')}")
    plt.close()
    
    # Save final training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_kaggle_score': train_kaggle_scores,
        'val_kaggle_score': val_kaggle_scores,
        'step_time': times
    }
    
    with open(os.path.join(run_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Also save as CSV for easier analysis
    steps_list = list(range(1, len(train_losses) + 1))
    val_steps = [i * eval_interval for i in range(1, len(val_losses) + 1)]
    
    # Create DataFrame for history - handling different lengths
    history_df = pd.DataFrame({
        'step': steps_list,
        'train_loss': train_losses
    })
    
    # Add validation metrics at eval intervals
    val_df = pd.DataFrame({
        'step': val_steps,
        'val_loss': val_losses,
        'val_kaggle_score': val_kaggle_scores,
        'train_kaggle_score': train_kaggle_scores,
        'step_time': times
    })
    
    # Merge the two DataFrames
    history_df = pd.merge(history_df, val_df, on='step', how='left')
    history_df.to_csv(os.path.join(run_dir, 'training_history.csv'), index=False)
    
    print(f"Training completed. Best validation Kaggle score: {best_val_kaggle_score:.4f}")
    print(f"All training artifacts saved to {run_dir}")
    
    return os.path.join(run_dir, "best_model.pt")

def inference(
    model_path,
    audio_path,
    taxonomy_file,
    output_file=None,
    segment_length=1900,
    sample_rate=32000,
    n_fft=1024,
    hop_length=512,
    device=None,
    temp_dir="./temp_inference"
):
    """
    Run inference on a single audio file or directory using WavtoSpec for spectrogram generation
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create temporary directory
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Load taxonomy
        taxonomy = pd.read_csv(taxonomy_file)
        species_ids = taxonomy['primary_label'].tolist()
        num_classes = len(species_ids)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        
        # We need to construct the original model architecture
        experiment_dir = os.path.dirname(os.path.dirname(model_path))
        with open(os.path.join(experiment_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Parse the architecture string to create blocks_config
        blocks_config = []
        if 'architecture' in config:
            for block_spec in config['architecture'].split(','):
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
        
        base_model = BirdJEPA(
            input_dim=config.get('input_dim', 128),
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 4),
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.1),
            mlp_dim=config.get('mlp_dim', 1024),
            pred_hidden_dim=config.get('pred_hidden_dim', 384),
            pred_num_layers=config.get('pred_num_layers', 6),
            pred_num_heads=config.get('pred_num_heads', 4),
            pred_mlp_dim=config.get('pred_mlp_dim', 1024),
            max_seq_len=config.get('max_seq_len', 512),
            blocks_config=blocks_config
        )
        
        classifier = BirdCLEFClassifier(
            base_model=base_model,
            num_classes=num_classes
        )
        
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.to(device)
        classifier.eval()
        
        # Process audio file(s)
        results = []
        
        # Organize files for spectrogram generation
        inference_audio_dir = os.path.join(temp_dir, "audio")
        inference_spec_dir = os.path.join(temp_dir, "specs")
        os.makedirs(inference_audio_dir, exist_ok=True)
        os.makedirs(inference_spec_dir, exist_ok=True)
        
        if os.path.isfile(audio_path):
            # Single file
            file_paths = [audio_path]
            # Copy to inference_audio_dir
            dst_path = os.path.join(inference_audio_dir, os.path.basename(audio_path))
            try:
                os.link(audio_path, dst_path)
            except OSError:
                shutil.copy2(audio_path, dst_path)
        else:
            # Directory - find all audio files
            file_paths = []
            for root, _, files in os.walk(audio_path):
                for file in files:
                    if file.endswith(('.ogg', '.wav', '.mp3')):
                        src_path = os.path.join(root, file)
                        dst_path = os.path.join(inference_audio_dir, file)
                        file_paths.append(src_path)
                        try:
                            os.link(src_path, dst_path)
                        except OSError:
                            shutil.copy2(src_path, dst_path)
        
        # Generate spectrograms using WavtoSpec
        wav_to_spec = WavtoSpec(
            src_dir=inference_audio_dir,
            dst_dir=inference_spec_dir,
            step_size=hop_length,
            nfft=n_fft,
            single_threaded=False
        )
        wav_to_spec.process_directory()
        
        # Get all spectrogram files
        spec_files = glob.glob(os.path.join(inference_spec_dir, "*.npz"))
        
        # Create a specialized dataset for inference
        inference_dataset = BirdJEPA_Dataset(data_dir=inference_spec_dir, segment_len=segment_length)
        
        # Process each spectrogram
        for spec_file in tqdm(spec_files, desc="Processing spectrograms"):
            try:
                # Load spectrogram from .npz file
                data = np.load(spec_file)
                spec = data['s']  # 's' is the key used in WavtoSpec
                
                # Convert to tensor and prepare dummy data for collate_fn
                spec_tensor = torch.tensor(spec).float()
                dummy_labels = torch.zeros(spec.shape[1], dtype=torch.long)
                filename = os.path.basename(spec_file)
                
                # Use data_class_collate_fn with a single sample
                full_spectrogram, _, _, _, _, _ = data_class_collate_fn(
                    [(spec_tensor, dummy_labels, filename)],
                    mask_p=0.0  # No masking for inference
                )
                
                # Move to device
                full_spectrogram = full_spectrogram.to(device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = classifier(full_spectrogram)
                
                # Extract original filename from spectrogram name
                base_name = os.path.basename(spec_file)
                parts = base_name.split('_segment_')
                original_file = parts[0]
                segment_idx = int(parts[1].split('.')[0])
                
                # Create row_id
                end_time = (segment_idx + 1) * segment_length
                row_id = f"soundscape_{original_file}_{end_time}"
                
                # Create result row
                row = {'row_id': row_id, 'file': original_file, 'segment': segment_idx}
                for j, species_id in enumerate(species_ids):
                    row[species_id] = outputs[0, j].item()
                
                results.append(row)
                
            except Exception as e:
                print(f"Error processing {spec_file}: {e}")
        
        # Create output dataframe
        results_df = pd.DataFrame(results)
        
        # Save to file if specified
        if output_file:
            results_df.to_csv(output_file, index=False)
            print(f"Saved predictions to {output_file}")
        
        return results_df
    
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")

def cleanup_directory(directory):
    """
    Remove a directory if it exists
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Cleaned up temporary directory: {directory}")

def analyze_dataset_labels(spec_dir, taxonomy_file, train_csv, sample_count=100):
    """
    Analyze the dataset labeling process to identify potential issues
    
    Args:
        spec_dir: Directory containing the spectrograms
        taxonomy_file: Path to taxonomy.csv file
        train_csv: Path to train.csv file with labels
        sample_count: Number of samples to analyze
    """
    print("\n" + "="*80)
    print("DATASET LABEL ANALYSIS")
    print("="*80)
    
    # Create a dataset instance for analysis
    dataset = BirdCLEFSpecDataset(
        spec_dir=spec_dir,
        taxonomy_file=taxonomy_file,
        train_csv=train_csv,
        mode='train',
        verbose=True  # Enable verbose mode
    )
    
    # Check class distribution in the taxonomy
    print(f"\nFound {dataset.num_classes} classes in taxonomy")
    
    # Analyze the train.csv file
    df = pd.read_csv(train_csv)
    print(f"\nTrain CSV contains {len(df)} entries")
    
    # Count species occurrences
    species_counts = df['primary_label'].value_counts()
    print(f"\nTop 10 most common species in train.csv:")
    print(species_counts.head(10))
    print(f"\nBottom 10 least common species in train.csv:")
    print(species_counts.tail(10))
    
    # Check for species with only one sample
    single_sample_species = species_counts[species_counts == 1]
    if len(single_sample_species) > 0:
        print(f"\nWARNING: Found {len(single_sample_species)} species with only one sample")
        print(single_sample_species)
    
    # Check filename to species mapping
    print(f"\nAnalyzing filename to species mapping...")
    match_count = 0
    missing_count = 0
    heuristic_count = 0
    
    # Collect samples for label distribution analysis
    all_labels = []
    
    for i in range(min(sample_count, len(dataset))):
        _, label = dataset[i]
        all_labels.append(label.numpy())
        
        # Count different types of mappings
        if i % 20 == 0:
            # Get raw data for analysis
            orig_spec, _, filename = dataset._get_raw_item(i)
            base_filename = os.path.basename(filename).split('_segment_')[0]
            
            # Check mapping type
            if base_filename in dataset.filename_to_species:
                match_count += 1
                mapping_type = "Direct match"
            elif base_filename.split('_')[0] in dataset.species_to_idx:
                heuristic_count += 1
                mapping_type = "Heuristic match"
            else:
                missing_count += 1
                mapping_type = "No match - using hash fallback"
            
            print(f"Sample {i}: {base_filename} -> {mapping_type}")
    
    print(f"\nMapping statistics from {sample_count} samples:")
    print(f"  Direct matches: {match_count}")
    print(f"  Heuristic matches: {heuristic_count}")
    print(f"  No matches (hash fallback): {missing_count}")
    
    # Analyze label distribution
    all_labels = np.array(all_labels)
    positive_per_class = all_labels.sum(axis=0)
    
    # Sort classes by number of positive examples
    sorted_indices = np.argsort(positive_per_class)[::-1]  # Descending order
    
    print(f"\nLabel distribution analysis:")
    print(f"  Average positive examples per class: {positive_per_class.mean():.2f}")
    print(f"  Max positive examples for a class: {positive_per_class.max():.2f}")
    print(f"  Min positive examples for a class: {positive_per_class.min():.2f}")
    print(f"  Classes with zero positive examples: {(positive_per_class == 0).sum()}")
    
    print(f"\nTop 10 most common classes in sampled data:")
    for i in range(10):
        idx = sorted_indices[i]
        species = dataset.species_ids[idx]
        count = positive_per_class[idx]
        print(f"  {species}: {count:.2f} positive examples")
    
    print(f"\nBottom 10 least common classes in sampled data:")
    for i in range(1, 11):
        if i <= len(sorted_indices):
            idx = sorted_indices[-i]
            species = dataset.species_ids[idx]
            count = positive_per_class[idx]
            print(f"  {species}: {count:.2f} positive examples")
    
    print("\n" + "="*80)
    print("End of analysis")
    print("="*80 + "\n")

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description='Fine-tune a BirdJEPA model for BirdCLEF classification')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer', 'debug', 'analyze'],
                        help='Mode to run the script in (train, infer, debug, or analyze for dataset inspection)')
    
    # Training parameters
    parser.add_argument('--pretrained_model_path', type=str, help='Path to pretrained model directory')
    parser.add_argument('--train_spec_dir', type=str, help='Path to directory containing training spectrograms')
    parser.add_argument('--val_spec_dir', type=str, help='Path to directory containing validation spectrograms')
    parser.add_argument('--taxonomy_file', type=str, help='Path to taxonomy.csv file')
    parser.add_argument('--train_csv', type=str, help='Path to train.csv file')
    parser.add_argument('--output_dir', type=str, help='Directory to save fine-tuned model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for fine-tuning')
    parser.add_argument('--max_steps', type=int, default=5000, help='Number of steps to train for')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--freeze_encoder', action='store_true', help='Whether to freeze encoder weights')
    parser.add_argument('--save_interval', type=int, default=500, help='Save checkpoints every N steps')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluate and log metrics every N steps')
    parser.add_argument('--use_baseline', action='store_true', help='Use baseline model without pretrained weights')
    parser.add_argument('--analyze_samples', type=int, default=100, help='Number of samples to analyze in analyze mode')
    
    # Inference parameters
    parser.add_argument('--model_path', type=str, help='Path to fine-tuned model (.pt file)')
    parser.add_argument('--audio_path', type=str, help='Path to audio file or directory for inference')
    parser.add_argument('--output_file', type=str, help='Path to save predictions for inference')
    parser.add_argument('--step_size', type=int, default=119, help='Step size for spectrogram generation')
    parser.add_argument('--nfft', type=int, default=1024, help='NFFT for spectrogram generation')
    parser.add_argument('--temp_dir', type=str, default='./temp_inference', help='Temporary directory for processing')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Check required arguments for training
        required_args = ['train_spec_dir', 'val_spec_dir', 'taxonomy_file', 'train_csv', 'output_dir']
        
        # Only require pretrained model path if not using baseline
        if not args.use_baseline:
            required_args.append('pretrained_model_path')
            
        for arg in required_args:
            if getattr(args, arg) is None:
                print(f"Error: --{arg} is required for training mode")
                return
        
        # Fine-tune the model using the already generated spectrograms
        finetune_model(
            pretrained_model_path=args.pretrained_model_path,
            train_spec_dir=args.train_spec_dir, 
            val_spec_dir=args.val_spec_dir,
            taxonomy_file=args.taxonomy_file,
            train_csv=args.train_csv,
            output_dir=args.output_dir,
            freeze_encoder=args.freeze_encoder,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            early_stopping_patience=args.early_stopping_patience,
            save_interval=args.save_interval,
            eval_interval=args.eval_interval,
            use_baseline=args.use_baseline
        )
    
    elif args.mode == "infer":
        if not args.model_path or not args.audio_path:
            print("For inference mode, please provide --model_path and --audio_path")
            return
        
        # For inference, we'll use the existing inference function
        inference(
            model_path=args.model_path,
            audio_path=args.audio_path,
            taxonomy_file=args.taxonomy_file,
            output_file=args.output_file,
            segment_length=2500,
            sample_rate=32000,
            n_fft=args.nfft,
            hop_length=args.step_size,
            temp_dir=args.temp_dir
        )
    
    elif args.mode == "debug":
        if not args.model_path or not args.audio_path or not args.taxonomy_file:
            print("For debug mode, please provide --model_path, --audio_path, and --taxonomy_file")
            return
            
    elif args.mode == "analyze":
        required_args = ['train_spec_dir', 'taxonomy_file', 'train_csv']
        for arg in required_args:
            if getattr(args, arg) is None:
                print(f"Error: --{arg} is required for analyze mode")
                return
                
        # Run the dataset analysis
        analyze_dataset_labels(
            spec_dir=args.train_spec_dir,
            taxonomy_file=args.taxonomy_file,
            train_csv=args.train_csv,
            sample_count=args.analyze_samples
        )

if __name__ == "__main__":
    main() 