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

# Import from project modules
from model import BirdJEPA
from utils import load_model
from spectrogram_generator import WavtoSpec
from data_class import BirdJEPA_Dataset, collate_fn as data_class_collate_fn

class BirdCLEFDataset(Dataset):
    """
    Dataset class for BirdCLEF 2025 competition data
    """
    def __init__(
        self,
        data_path,
        taxonomy_file,
        train_csv=None,
        mode='train',
        segment_length=5,  # 5 second segments
        sample_rate=32000,  # 32kHz sample rate as per competition
        n_fft=1024,
        hop_length=512,
        augment=True,
        use_cached_spectrograms=False,
        cache_dir=None,
        max_samples_per_species=None
    ):
        self.data_path = data_path
        self.mode = mode
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augment = augment and mode == 'train'
        self.use_cached_spectrograms = use_cached_spectrograms
        self.cache_dir = cache_dir
        self.max_samples_per_species = max_samples_per_species
        
        # Load taxonomy (species list)
        self.taxonomy = pd.read_csv(taxonomy_file)
        self.species_ids = self.taxonomy['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        
        # Create mapping from species ID to index
        self.species_to_idx = {s: i for i, s in enumerate(self.species_ids)}
        
        # Load file paths and labels based on mode
        if mode == 'train' or mode == 'val':
            # Load train.csv
            train_df = pd.read_csv(train_csv)
            
            # If validation mode, use a consistent subset of data
            if mode == 'val':
                # Use 10% of each species data for validation
                train_df = self._split_train_val(train_df, val_fraction=0.1)
                
            self.file_paths = []
            self.labels = []
            
            # For each row in train_df, get the file path and primary label
            for _, row in train_df.iterrows():
                species_id = row['primary_label']
                filename = row['filename']
                
                # Construct the full file path - the filename in CSV already contains species_id/file.ogg
                filepath = os.path.join(self.data_path, 'train_audio', filename)
                
                if os.path.exists(filepath):
                    self.file_paths.append(filepath)
                    
                    # One-hot encoding for primary label
                    label = np.zeros(self.num_classes)
                    label[self.species_to_idx[species_id]] = 1
                    
                    # Add secondary labels if present
                    if 'secondary_labels' in row and not pd.isna(row['secondary_labels']):
                        try:
                            if isinstance(row['secondary_labels'], str):
                                secondary_labels = eval(row['secondary_labels'])
                            else:
                                secondary_labels = row['secondary_labels']
                            
                            for sec_label in secondary_labels:
                                if sec_label in self.species_to_idx and sec_label != '':
                                    label[self.species_to_idx[sec_label]] = 1
                        except (SyntaxError, ValueError) as e:
                            print(f"Error processing secondary labels for {filename}: {e}")
                    
                    self.labels.append(label)
                else:
                    print(f"File not found: {filepath}")
            
            # If max_samples_per_species is set, limit the number of samples
            if self.max_samples_per_species is not None:
                self._limit_samples_per_species()
                
        elif mode == 'soundscape':
            # For unlabeled soundscapes (no labels)
            soundscape_dir = os.path.join(self.data_path, 'train_soundscapes')
            self.file_paths = [
                os.path.join(soundscape_dir, f) 
                for f in os.listdir(soundscape_dir) 
                if f.endswith('.ogg')
            ]
            self.labels = [np.zeros(self.num_classes) for _ in self.file_paths]
        
        print(f"Loaded {len(self.file_paths)} files for {mode} mode")

    def _split_train_val(self, df, val_fraction=0.1):
        """Split dataframe into train and validation sets stratified by species"""
        # If we're in validation mode, return the validation subset
        val_indices = []
        
        # Group by species ID
        for species_id in df['primary_label'].unique():
            species_df = df[df['primary_label'] == species_id]
            indices = species_df.index.tolist()
            
            # Randomly select validation indices
            num_val = max(1, int(len(indices) * val_fraction))
            random.seed(42)  # For reproducibility
            val_indices.extend(random.sample(indices, num_val))
        
        if self.mode == 'val':
            return df.loc[val_indices]
        else:
            return df.drop(val_indices)

    def _limit_samples_per_species(self):
        """Limit the number of samples per species for balanced training"""
        by_species = {}
        
        # Group samples by species
        for i, label in enumerate(self.labels):
            species_idx = np.argmax(label)
            if species_idx not in by_species:
                by_species[species_idx] = []
            by_species[species_idx].append(i)
        
        # Limit each species to max_samples_per_species
        new_file_paths = []
        new_labels = []
        
        for species_idx, indices in by_species.items():
            if len(indices) > self.max_samples_per_species:
                indices = random.sample(indices, self.max_samples_per_species)
            
            for i in indices:
                new_file_paths.append(self.file_paths[i])
                new_labels.append(self.labels[i])
        
        self.file_paths = new_file_paths
        self.labels = new_labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Check cache first if using cached spectrograms
        if self.use_cached_spectrograms and self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{os.path.basename(file_path)}.npz")
            if os.path.exists(cache_path):
                # Load cached spectrogram
                cached_data = np.load(cache_path)
                spec = cached_data['s']  # Matches the key used in WavtoSpec
                
                # Handle different segment selection based on mode
                if self.mode == 'train':
                    # For training, randomly select a segment
                    if spec.shape[1] > self.segment_length * self.sample_rate // self.hop_length:
                        start = random.randint(0, spec.shape[1] - self.segment_length * self.sample_rate // self.hop_length)
                        spec = spec[:, start:start + self.segment_length * self.sample_rate // self.hop_length]
                    else:
                        # Pad if necessary
                        target_length = self.segment_length * self.sample_rate // self.hop_length
                        if spec.shape[1] < target_length:
                            padding = target_length - spec.shape[1]
                            spec = np.pad(spec, ((0, 0), (0, padding)))
                
                return torch.tensor(spec).float(), torch.tensor(label).float()
        
        # Process audio file if not cached or cache not found
        try:
            # Load the audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            
            # Handle audio shorter than segment_length
            if len(audio) < self.segment_length * self.sample_rate:
                # Pad with zeros
                padding = self.segment_length * self.sample_rate - len(audio)
                audio = np.pad(audio, (0, padding))
            
            # For training, randomly select a segment
            if self.mode == 'train':
                if len(audio) > self.segment_length * self.sample_rate:
                    start = random.randint(0, len(audio) - self.segment_length * self.sample_rate)
                    audio = audio[start:start + self.segment_length * self.sample_rate]
            
            # For validation or test, use the first segment
            elif len(audio) > self.segment_length * self.sample_rate:
                audio = audio[:self.segment_length * self.sample_rate]
            
            # Apply the same preprocessing as in WavtoSpec
            # High-pass filter
            from scipy.signal import ellip, filtfilt
            b, a = ellip(5, 0.2, 40, 500/(self.sample_rate/2), 'high')
            audio = filtfilt(b, a, audio)
            
            # Compute STFT using the same parameters as WavtoSpec
            Sxx = librosa.stft(audio.astype(float), n_fft=self.n_fft, hop_length=self.hop_length, window='hann')
            
            # Convert to dB scale as in WavtoSpec
            Sxx_log = librosa.amplitude_to_db(np.abs(Sxx), ref=np.max)
            
            # Cache the spectrogram if needed
            if self.use_cached_spectrograms and self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                cache_path = os.path.join(self.cache_dir, f"{os.path.basename(file_path)}.npz")
                np.savez_compressed(
                    cache_path,
                    s=Sxx_log,
                    vocalization=np.ones(Sxx_log.shape[1], dtype=int),  # All frames considered as vocalization
                    labels=np.zeros(Sxx_log.shape[1], dtype=int)  # No per-frame labels
                )
            
            return torch.tensor(Sxx_log).float(), torch.tensor(label).float()
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Return zeros as fallback
            spec = np.zeros((self.n_fft // 2 + 1, self.segment_length * self.sample_rate // self.hop_length))
            return torch.tensor(spec).float(), torch.tensor(label).float()

def collate_fn(batch):
    """Custom collate function to handle variable length spectrograms"""
    specs = []
    labels = []
    
    # Find max spectrogram length in this batch
    max_time_len = 0
    for spec, label in batch:
        max_time_len = max(max_time_len, spec.shape[1])
    
    # Pad each spectrogram to the max length in this batch
    for spec, label in batch:
        if spec.shape[1] < max_time_len:
            # Pad the time dimension to match max_time_len
            padded_spec = torch.nn.functional.pad(spec, (0, max_time_len - spec.shape[1]))
            specs.append(padded_spec)
        else:
            specs.append(spec)
        labels.append(label)
    
    # Stack all tensors
    specs = torch.stack(specs)
    labels = torch.stack(labels)
    
    return specs, labels

class BirdCLEFClassifier(nn.Module):
    """
    Adaptation of BirdJEPA model for BirdCLEF classification
    """
    def __init__(
        self, 
        base_model,
        num_classes,
        freeze_encoder=True,
        classifier_hidden_dim=512
    ):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        
        # Get hidden dimension from base model
        hidden_dim = self.base_model.hidden_dim
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.base_model.context_encoder.parameters():
                param.requires_grad = False
        
        # Create classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(classifier_hidden_dim, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (B, F, T) - spectrogram from STFT
        
        # Add channel dimension for CNN
        x = x.unsqueeze(1)  # Add channel dim: (B, 1, F, T)
        
        # Get embeddings from base model
        with torch.no_grad() if all(p.requires_grad == False for p in self.base_model.context_encoder.parameters()) else torch.enable_grad():
            try:
                # Extract features using the feature extractor
                features = self.base_model.context_encoder.feature_extractor(x)
                
                # 1. Get dimensions
                B, C, Freq, T = features.shape
                
                # 2. Flatten the frequency dimension
                features = features.view(B, C * Freq, T)  # (B, C*Freq, T)
                
                # 3. Transpose to match the expected input format for the projection
                features = features.transpose(1, 2)  # (B, T, C*Freq)
                
                # Handle dimension mismatch
                # The input_proj might expect a different dimension than what we have
                input_proj_weight = self.base_model.context_encoder.input_proj.weight
                expected_dim = input_proj_weight.shape[1]
                actual_dim = features.shape[2]
                
                if expected_dim != actual_dim:
                    if expected_dim < actual_dim:
                        # Reduce dimension using average pooling
                        features_t = features.transpose(1, 2)  # (B, C*Freq, T)
                        pooled = torch.nn.functional.adaptive_avg_pool1d(features_t, expected_dim)
                        features = pooled.transpose(1, 2)  # Back to (B, T, expected_dim)
                    else:
                        # This case is unlikely but handle it by padding
                        padding = torch.zeros(B, features.shape[1], expected_dim - actual_dim, device=features.device)
                        features = torch.cat([features, padding], dim=2)
                
                # 4. Now we can directly use the input_proj from the context_encoder
                embeddings = self.base_model.context_encoder.input_proj(features)
                
                # 5. Apply positional encoding and dropout as the original encoder does
                embeddings = self.base_model.context_encoder.dropout(embeddings)
                embeddings = self.base_model.context_encoder.pos_enc(embeddings)
                
                # 6. Process through transformer blocks
                for block in self.base_model.context_encoder.attention_blocks:
                    embeddings = block(embeddings)
                
                # Average pooling over sequence dimension
                pooled = embeddings.mean(dim=1)  # (B, hidden_dim)
                
            except Exception as e:
                print(f"Error in forward pass: {e}")
                # Fallback to random embedding
                hidden_dim = self.base_model.hidden_dim
                pooled = torch.zeros((x.shape[0], hidden_dim), device=x.device)
        
        # Classification
        output = self.classifier(pooled)
        
        return output

def create_train_val_split(data_path, train_csv, val_percentage, temp_dir, taxonomy_file, max_files=None):
    """
    Create a train/validation split based on the BirdCLEF dataset.
    Similar to how it's done in pretrain.sh.
    
    Args:
        data_path: Path to the BirdCLEF dataset directory
        train_csv: Path to the train CSV file
        val_percentage: Percentage of data to use for validation (0-100)
        temp_dir: Directory to store temporary files
        taxonomy_file: Path to the taxonomy CSV file
        max_files: Maximum number of files to process (for testing)
        
    Returns:
        Tuple of (train_dir, val_dir, train_file_list, val_file_list)
    """
    print(f"Creating train/val split with {val_percentage}% validation data")
    
    # Create temporary directories
    os.makedirs(temp_dir, exist_ok=True)
    train_audio_dir = os.path.join(temp_dir, "train_audio")
    val_audio_dir = os.path.join(temp_dir, "val_audio")
    train_dir = os.path.join(temp_dir, "train_specs")
    val_dir = os.path.join(temp_dir, "val_specs")
    
    os.makedirs(train_audio_dir, exist_ok=True)
    os.makedirs(val_audio_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Load train.csv
    train_df = pd.read_csv(train_csv)
    
    # If max_files is set, limit the number of samples
    if max_files is not None:
        print(f"Limiting to {max_files} files for testing purposes")
        # Ensure we get data from each species if possible
        unique_species = train_df['primary_label'].unique()
        max_per_species = max(1, max_files // len(unique_species))
        
        # Select samples from each species
        limited_df = []
        for species in unique_species:
            species_samples = train_df[train_df['primary_label'] == species].sample(
                min(max_per_species, sum(train_df['primary_label'] == species)),
                random_state=42
            )
            limited_df.append(species_samples)
        
        # Combine and limit to max_files if needed
        train_df = pd.concat(limited_df)
        if len(train_df) > max_files:
            train_df = train_df.sample(max_files, random_state=42)
        
        print(f"Limited dataset to {len(train_df)} files")
    
    # Load taxonomy
    taxonomy = pd.read_csv(taxonomy_file)
    
    # Split by species to ensure stratification
    train_files = []
    val_files = []
    
    # Set a random seed for reproducibility
    random.seed(42)
    
    # Group by species
    species_groups = {}
    for species_id in taxonomy['primary_label'].tolist():
        species_groups[species_id] = []
    
    # Assign each file to its species group
    for _, row in train_df.iterrows():
        species_id = row['primary_label']
        filename = row['filename']
        src_path = os.path.join(data_path, 'train_audio', filename)
        
        if os.path.exists(src_path):
            species_groups[species_id].append((filename, src_path))
    
    # Calculate total number of validation files needed
    total_files = sum(len(files) for files in species_groups.values())
    total_val_files_needed = int(total_files * val_percentage / 100)
    
    # Ensure we have at least some validation files
    if total_val_files_needed == 0 and total_files > 0:
        total_val_files_needed = max(1, int(total_files * 0.1))  # At least 10% or 1 file
        print(f"Adjusted validation percentage to ensure at least {total_val_files_needed} validation files")
    
    # Track counts
    val_files_assigned = 0
    
    # Now assign files to train/val sets by species
    for species_id, files in species_groups.items():
        if not files:
            continue
        
        # Calculate number of validation examples for this species
        species_val_count = max(1, int(len(files) * val_percentage / 100))
        
        # For species with only 1 file, include in training if we already have enough validation files
        if len(files) == 1 and val_files_assigned >= total_val_files_needed / 2:
            species_val_count = 0
        elif len(files) == 1:
            species_val_count = 1
        # Otherwise ensure at least 1 file is kept for training if there are multiple files
        elif len(files) > 1:
            species_val_count = min(species_val_count, len(files) - 1)
        
        # Randomly select validation files
        val_indices = set(random.sample(range(len(files)), species_val_count))
        val_files_assigned += species_val_count
        
        # Process files
        for i, (filename, src_path) in enumerate(files):
            # Create directory structure if needed
            species_folder = os.path.dirname(filename)
            
            if i in val_indices:
                # Validation file
                dst_dir = os.path.join(val_audio_dir, species_folder)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(val_audio_dir, filename)
                val_files.append(dst_path)
            else:
                # Training file
                dst_dir = os.path.join(train_audio_dir, species_folder)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(train_audio_dir, filename)
                train_files.append(dst_path)
                
            # Create hard links instead of copying to save space
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            if not os.path.exists(dst_path):
                try:
                    os.link(src_path, dst_path)
                except OSError:
                    # Fallback to copy if hard linking fails
                    shutil.copy2(src_path, dst_path)
    
    # Write file lists
    train_file_list = os.path.join(temp_dir, "train_files.txt")
    val_file_list = os.path.join(temp_dir, "val_files.txt")
    
    with open(train_file_list, 'w') as f:
        for file_path in train_files:
            f.write(f"{file_path}\n")
            
    with open(val_file_list, 'w') as f:
        for file_path in val_files:
            f.write(f"{file_path}\n")
    
    print(f"Created {len(train_files)} training files and {len(val_files)} validation files")
    
    return train_audio_dir, val_audio_dir, train_dir, val_dir, train_file_list, val_file_list

def generate_spectrograms(audio_dir, spec_dir, step_size, nfft, multi_thread=True, song_detection_json_path=None, max_random_files=None):
    """
    Generate spectrograms using the spectrogram_generator.py script.
    This mimics how it's done in pretrain.sh.
    
    Args:
        audio_dir: Directory containing audio files
        spec_dir: Directory to save spectrograms
        step_size: Step size for spectrogram generation
        nfft: NFFT value for spectrogram generation
        multi_thread: Whether to use multi-threading
        song_detection_json_path: Optional path to song detection JSON
        max_random_files: Maximum number of random files to process (for testing)
        
    Returns:
        Exit code of the subprocess
    """
    print(f"Generating spectrograms from {audio_dir} to {spec_dir}")
    
    # Determine single_threaded parameter (inverted from multi_thread)
    single_threaded = 'true' if not multi_thread else 'false'
    
    # Set song_detection_json_path to "None" if None
    if song_detection_json_path is None:
        song_detection_json_path = "None"
    
    # Use the same Python executable that's running this script
    python_executable = sys.executable
    print(f"Using Python executable: {python_executable}")
    
    # Build command
    cmd = [
        python_executable, 'src/spectrogram_generator.py',
        '--src_dir', audio_dir,
        '--dst_dir', spec_dir,
        '--song_detection_json_path', song_detection_json_path,
        '--step_size', str(step_size),
        '--nfft', str(nfft),
        '--single_threaded', single_threaded
    ]
    
    # Add max_random_files if specified
    if max_random_files is not None:
        cmd.extend(['--generate_random_files_number', str(max_random_files)])
    
    # Execute command
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print(process.stdout)
    
    if process.returncode != 0:
        print(f"Error generating spectrograms: {process.stderr}")
    else:
        print(f"Successfully generated spectrograms in {spec_dir}")
    
    return process.returncode

class BirdCLEFSpecDataset(Dataset):
    """
    Dataset class for BirdCLEF 2025 competition data using pre-generated spectrograms
    """
    def __init__(
        self,
        spec_dir,
        taxonomy_file,
        train_csv=None,
        mode='train',
        max_samples_per_species=None,
        segment_len=None  # Use segment_len for compatibility with BirdJEPA_Dataset
    ):
        """
        Initialize the dataset
        
        Args:
            spec_dir: Directory containing spectrogram files (.npz)
            taxonomy_file: Path to taxonomy.csv file
            train_csv: Path to train.csv file (needed to map filenames to species)
            mode: 'train' or 'val'
            max_samples_per_species: Maximum number of samples per species (for balanced training)
            segment_len: Used for compatibility with BirdJEPA_Dataset (if None, will use variable lengths)
        """
        self.spec_dir = spec_dir
        self.mode = mode
        self.segment_len = segment_len
        
        # Load taxonomy file
        taxonomy = pd.read_csv(taxonomy_file)
        self.species_ids = taxonomy['primary_label'].astype(str).tolist()
        self.species_to_idx = {species_id: i for i, species_id in enumerate(self.species_ids)}
        self.num_classes = len(self.species_ids)
        
        # Load train.csv to create filename->species mapping
        self.filename_to_species = {}
        if train_csv is not None:
            train_df = pd.read_csv(train_csv)
            for _, row in train_df.iterrows():
                # Extract species_id and filename from the full path
                parts = row['filename'].split('/')
                if len(parts) == 2:
                    species_id, filename = parts
                    # Remove file extension
                    filename = os.path.splitext(filename)[0]
                    self.filename_to_species[filename] = species_id
        
        # Find all spectrogram files
        spec_files = glob.glob(os.path.join(spec_dir, '*.npz'))
        filtered_spec_files = []
        self.labels = []
        
        for spec_file in spec_files:
            # Extract base filename without extension
            file_basename = os.path.basename(spec_file)
            # Remove segment suffix (e.g., _segment_0.npz) to get original file name
            file_prefix = file_basename.split('_segment_')[0]
            
            species_id = None
            
            # First, try to find from filename_to_species mapping (from train.csv)
            if file_prefix in self.filename_to_species:
                species_id = str(self.filename_to_species[file_prefix])
            
            # If not found, try other methods
            if species_id is None:
                # Try to match with a known species ID
                for s_id in self.species_ids:
                    if file_prefix.startswith(s_id):
                        species_id = s_id
                        break
            
            if species_id is None:
                # Try to extract from directory structure
                # The format should be {species_id}/{filename}
                parent_dir = os.path.basename(os.path.dirname(spec_file))
                if parent_dir in self.species_ids:
                    species_id = parent_dir
            
            if species_id is None:
                print(f"Warning: Could not determine species for {spec_file}")
                continue
            
            # Create one-hot encoding for the label
            label = np.zeros(self.num_classes)
            if species_id in self.species_to_idx:
                label[self.species_to_idx[species_id]] = 1
                filtered_spec_files.append(spec_file)
                self.labels.append(label)
            else:
                print(f"Warning: Unknown species ID: {species_id}")
        
        self.spec_files = filtered_spec_files
        
        # If max_samples_per_species is set, limit the number of samples
        if max_samples_per_species is not None:
            self._limit_samples_per_species(max_samples_per_species)
                
        print(f"Loaded {len(self.spec_files)} spectrogram files for {mode} mode")

    def _limit_samples_per_species(self, max_samples_per_species):
        """Limit the number of samples per species for balanced training"""
        by_species = {}
        
        # Group samples by species
        for i, label in enumerate(self.labels):
            species_idx = np.argmax(label)
            if species_idx not in by_species:
                by_species[species_idx] = []
            by_species[species_idx].append(i)
        
        # Limit each species to max_samples_per_species
        new_spec_files = []
        new_labels = []
        
        for species_idx, indices in by_species.items():
            if len(indices) > max_samples_per_species:
                indices = random.sample(indices, max_samples_per_species)
            
            for i in indices:
                new_spec_files.append(self.spec_files[i])
                new_labels.append(self.labels[i])
        
        self.spec_files = new_spec_files
        self.labels = new_labels
        print(f"Limited dataset to max {max_samples_per_species} samples per species, total samples: {len(self.spec_files)}")

    def __len__(self):
        return len(self.spec_files)

    def __getitem__(self, idx):
        spec_file = self.spec_files[idx]
        label = self.labels[idx]
        
        try:
            # Load spectrogram from .npz file
            data = np.load(spec_file)
            spec = data['s']  # 's' is the key used in WavtoSpec
            
            # For train mode, if segment_len is specified, select a random segment
            if self.mode == 'train' and self.segment_len is not None and spec.shape[1] > self.segment_len:
                start = random.randint(0, spec.shape[1] - self.segment_len)
                spec = spec[:, start:start + self.segment_len]
            
            # For validation mode, use the first segment if segment_len is specified
            elif self.mode == 'val' and self.segment_len is not None and spec.shape[1] > self.segment_len:
                spec = spec[:, :self.segment_len]
            
            # Otherwise, use the full spectrogram (data_class will handle padding/trimming)
            
            # Convert to tensor
            spec_tensor = torch.tensor(spec).float()
            label_tensor = torch.tensor(label).float()
            
            # For compatibility with BirdJEPA_Dataset, also return the filename
            filename = os.path.basename(spec_file)
            
            return spec_tensor, label_tensor, filename
            
        except Exception as e:
            print(f"Error loading {spec_file}: {e}")
            # Return zeros as fallback with a reasonable default size
            spec = np.zeros((513, 100))  # Default size: 513 frequency bins, 100 time bins
            return torch.tensor(spec).float(), torch.tensor(label).float(), "error.npz"

def classification_collate_fn(batch):
    """
    Adapt data_class_collate_fn for classification tasks
    We need to handle the 3-tuple return values from BirdCLEFSpecDataset
    and prepare them for data_class_collate_fn
    """
    # Unpack batch of (spec, label, filename)
    specs, labels, filenames = zip(*batch)
    
    # Find max spectrogram length in this batch
    max_time_len = max(spec.shape[1] for spec in specs)
    
    # Pad each spectrogram to the max length in this batch
    padded_specs = []
    for spec in specs:
        if spec.shape[1] < max_time_len:
            # Pad the time dimension to match max_time_len
            padding = torch.zeros((spec.shape[0], max_time_len - spec.shape[1]), dtype=spec.dtype)
            padded_spec = torch.cat([spec, padding], dim=1)
            padded_specs.append(padded_spec)
        else:
            padded_specs.append(spec)
    
    # Create dummy labels for data_class_collate_fn (we'll use our actual labels later)
    dummy_labels = [torch.zeros(spec.shape[1], dtype=torch.long) for spec in padded_specs]
    
    # Use data_class_collate_fn with a mask_p of 0 (no masking for classification)
    # The signature is: full_spec, target_spec, context_spec, labels, mask, filenames
    full_spectrogram, _, _, _, _, _ = data_class_collate_fn(
        list(zip(padded_specs, dummy_labels, filenames)),
        mask_p=0.0  # No masking for classification
    )
    
    # Stack the actual classification labels
    stacked_labels = torch.stack(labels)
    
    return full_spectrogram, stacked_labels

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
    epochs=30,
    early_stopping_patience=5,
    max_samples_per_species=None,
    device=None
):
    """
    Fine-tune a pretrained BirdJEPA model for BirdCLEF classification
    using pre-generated spectrograms
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pretrained model
    print(f"Loading pretrained model from {pretrained_model_path}")
    base_model, _, config = load_model(pretrained_model_path, return_checkpoint=True)
    
    # Save model config to output directory
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Create datasets
    print(f"Creating datasets")
    train_dataset = BirdCLEFSpecDataset(
        spec_dir=train_spec_dir,
        taxonomy_file=taxonomy_file,
        train_csv=train_csv,
        mode='train',
        max_samples_per_species=max_samples_per_species
    )
    
    val_dataset = BirdCLEFSpecDataset(
        spec_dir=val_spec_dir,
        taxonomy_file=taxonomy_file,
        train_csv=train_csv,
        mode='val'
    )
    
    # Create data loaders
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
    
    # Create classifier model
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
    
    # Training loop
    best_val_auc = 0
    early_stopping_counter = 0
    train_losses = []
    val_losses = []
    val_aucs = []
    
    for epoch in range(epochs):
        # Training
        classifier.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_data in progress_bar:
            # Unpack batch data - our classification_collate_fn returns (spectrograms, labels)
            specs, labels = batch_data
            
            # Move data to device
            specs = specs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = classifier(specs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        classifier.eval()
        val_loss = 0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch_data in progress_bar:
                # Unpack batch data - our classification_collate_fn returns (spectrograms, labels)
                specs, labels = batch_data
                
                # Move data to device
                specs = specs.to(device)
                labels = labels.to(device)
                
                outputs = classifier(specs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                
                # Collect outputs and labels for AUC calculation
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Compute validation loss and AUC
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        all_outputs = np.vstack(all_outputs)
        all_labels = np.vstack(all_labels)
        
        # Compute per-class AUC and average
        aucs = []
        for i in range(train_dataset.num_classes):
            # Only compute AUC if there are positive examples
            if np.sum(all_labels[:, i]) > 0:
                try:
                    auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
                    aucs.append(auc)
                except:
                    pass
        
        val_auc = np.mean(aucs) if aucs else 0
        val_aucs.append(val_auc)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        # Update learning rate scheduler
        scheduler.step(val_auc)
        
        # Check if this is the best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            early_stopping_counter = 0
            
            # Save model
            model_path = os.path.join(output_dir, f"best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, model_path)
            print(f"Saved best model with AUC {val_auc:.4f}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_auc': val_aucs
    }
    
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_aucs, label='Val AUC')
    plt.title('AUC')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    print(f"Training completed. Best validation AUC: {best_val_auc:.4f}")
    
    return os.path.join(output_dir, "best_model.pt")

def inference(
    model_path,
    audio_path,
    taxonomy_file,
    output_file=None,
    segment_length=5,
    sample_rate=32000,
    n_fft=1024,
    hop_length=512,
    device=None,
    temp_dir="./temp_inference"
):
    """
    Run inference on a single audio file or directory
    
    Args:
        model_path: Path to the fine-tuned model
        audio_path: Path to audio file or directory
        taxonomy_file: Path to the taxonomy CSV file
        output_file: Path to save predictions (optional)
        segment_length: Length of segments in seconds
        sample_rate: Audio sample rate
        n_fft: FFT size
        hop_length: Hop length for STFT
        device: Device to use for inference
        temp_dir: Temporary directory for processing
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
        
        # Generate spectrograms
        generate_spectrograms(
            audio_dir=inference_audio_dir,
            spec_dir=inference_spec_dir,
            step_size=hop_length,
            nfft=n_fft,
            multi_thread=True,
            song_detection_json_path=None,
            max_random_files=None
        )
        
        # Get all spectrogram files
        spec_files = glob.glob(os.path.join(inference_spec_dir, "*.npz"))
        
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

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description='Fine-tune a BirdJEPA model for BirdCLEF classification')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'],
                        help='Mode to run the script in (train or infer)')
    
    # Training parameters
    parser.add_argument('--pretrained_model_path', type=str, help='Path to pretrained model directory')
    parser.add_argument('--data_path', type=str, help='Path to BirdCLEF dataset root directory')
    parser.add_argument('--taxonomy_file', type=str, help='Path to taxonomy.csv file')
    parser.add_argument('--train_csv', type=str, help='Path to train.csv file')
    parser.add_argument('--output_dir', type=str, help='Directory to save fine-tuned model')
    parser.add_argument('--val_percentage', type=float, default=10.0, help='Percentage of data to use for validation')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for fine-tuning')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train for')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--max_samples_per_species', type=int, default=100, help='Maximum number of samples per species')
    parser.add_argument('--freeze_encoder', action='store_true', help='Whether to freeze the encoder weights')
    parser.add_argument('--multi_thread', action='store_true', help='Whether to use multi-threading for spectrogram generation')
    parser.add_argument('--step_size', type=int, default=119, help='Step size for spectrogram generation')
    parser.add_argument('--nfft', type=int, default=1024, help='NFFT for spectrogram generation')
    parser.add_argument('--temp_dir', type=str, default='./temp_finetuning', help='Temporary directory for processing')
    parser.add_argument('--max_files', type=int, help='Maximum number of files to process (for testing)')
    parser.add_argument('--max_random_files', type=int, help='Maximum number of random files to process for spectrograms (for testing)')
    parser.add_argument('--song_detection_json_path', type=str, default=None, help='Path to song detection JSON file')
    
    # Inference parameters
    parser.add_argument('--model_path', type=str, help='Path to fine-tuned model (.pt file)')
    parser.add_argument('--audio_path', type=str, help='Path to audio file or directory for inference')
    parser.add_argument('--output_file', type=str, help='Path to save predictions for inference')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Check required arguments for training
        required_args = ['pretrained_model_path', 'data_path', 'taxonomy_file', 'train_csv', 'output_dir']
        for arg in required_args:
            if getattr(args, arg) is None:
                print(f"Error: --{arg} is required for training mode")
                return
        
        # Create train/val split
        temp_dir = args.temp_dir
        print(f"Creating train/val split with {args.val_percentage}% validation data")
        
        train_audio_dir, val_audio_dir, train_dir, val_dir, train_file_list, val_file_list = create_train_val_split(
            data_path=args.data_path,
            train_csv=args.train_csv,
            val_percentage=args.val_percentage,
            temp_dir=temp_dir,
            taxonomy_file=args.taxonomy_file,
            max_files=args.max_files
        )
        
        # Generate spectrograms for train/val sets
        train_ret = generate_spectrograms(
            train_audio_dir, 
            train_dir, 
            args.step_size, 
            args.nfft,
            multi_thread=args.multi_thread,
            song_detection_json_path=args.song_detection_json_path,
            max_random_files=args.max_random_files
        )
        
        val_ret = generate_spectrograms(
            val_audio_dir, 
            val_dir, 
            args.step_size, 
            args.nfft,
            multi_thread=args.multi_thread,
            song_detection_json_path=args.song_detection_json_path,
            max_random_files=args.max_random_files
        )
        
        # Check if any spectrograms were generated
        train_specs = glob.glob(os.path.join(train_dir, "*.npz"))
        val_specs = glob.glob(os.path.join(val_dir, "*.npz"))
        
        print(f"Generated {len(train_specs)} training spectrograms and {len(val_specs)} validation spectrograms")
        
        if len(train_specs) == 0 and len(val_specs) == 0:
            print("Error: No spectrograms were generated. Train: 0, Val: 0")
            print("Please check if the spectrogram_generator.py script is working correctly.")
            
            # Clean up
            cleanup_directory(temp_dir)
            return
        
        # Fine-tune the model
        finetune_model(
            pretrained_model_path=args.pretrained_model_path,
            train_spec_dir=train_dir,
            val_spec_dir=val_dir,
            taxonomy_file=args.taxonomy_file,
            train_csv=args.train_csv,
            output_dir=args.output_dir,
            freeze_encoder=args.freeze_encoder,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            early_stopping_patience=args.early_stopping_patience,
            max_samples_per_species=args.max_samples_per_species
        )
        
        # Clean up temporary directory
        cleanup_directory(temp_dir)
    
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
            segment_length=5,
            sample_rate=32000,
            n_fft=args.nfft,
            hop_length=args.step_size,
            temp_dir=args.temp_dir
        )

if __name__ == "__main__":
    main() 