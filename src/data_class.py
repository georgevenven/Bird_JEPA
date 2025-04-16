import torch
from torch.utils.data import Dataset
import random
import os
import numpy as np
import torchaudio
import time
from pathlib import Path
from timing_utils import Timer, timed_operation, timing_stats

class BirdJEPA_Dataset(Dataset):
    def __init__(self, data_dir, segment_len=50, verbose=False, infinite_dataset=True):
        self.data_dir = data_dir
        self.segment_len = segment_len
        self.verbose = verbose
        self.infinite_dataset = infinite_dataset
        # preload file paths into memory for fast access
        self.file_paths = [entry.path for entry in os.scandir(data_dir)]
        self.file_count = len(self.file_paths)
        
        # DEBUG: Print all file paths and their indices
        print(f"DEBUG: BirdJEPA_Dataset initialized with {self.file_count} files in {data_dir}")
        for i, path in enumerate(self.file_paths):
            print(f"DEBUG: File {i}: {path}")
        
        if self.verbose:
            print(f"Initialized dataset with {self.file_count} files.")
                
    def __len__(self):
        if self.infinite_dataset:
            # return a very large number to simulate an infinite dataset
            return int(1e5)
        else:
            return self.file_count
    
    def _get_random_file(self):
        """Get a random file path from the preloaded file paths."""
        idx = random.randint(0, self.file_count - 1)
        file_path = self.file_paths[idx]
        print(f"DEBUG: _get_random_file selected index {idx}: {file_path}")
        return file_path
    
    def _load_file(self, file_path):
        """Load file with memory mapping for efficient partial reads."""
        print(f"DEBUG: Loading file: {file_path}")
        return np.load(file_path, allow_pickle=True, mmap_mode='r')

    @timed_operation("data_loading")
    def __getitem__(self, idx):
        """Get a segment from a file."""
        print(f"DEBUG: __getitem__ called with idx={idx}, file_count={self.file_count}")
        
        if self.infinite_dataset:
            # For infinite datasets, use random file selection
            file_path = self._get_random_file()
            print(f"DEBUG: Infinite mode - random file selected: {file_path}")
        else:
            # For finite datasets, use deterministic file selection
            safe_idx = idx % self.file_count
            file_path = self.file_paths[safe_idx]
            print(f"DEBUG: Finite mode - using idx={idx}, safe_idx={safe_idx}, file={file_path}")
            print(f"DEBUG: Full file list: {self.file_paths}")
        
        # Load with memory mapping
        data = self._load_file(file_path)
        # Get the spectrogram and remove the final timebin
        spec = data['s'][:, :-1]  # Memory-mapped array with last timebin removed
        ground_truth_labels = data['labels'] # Memory-mapped array

        F, T = spec.shape
        
        if self.verbose:
            print(f"Loaded file: {file_path}")
            print(f"Original spectrogram shape: {spec.shape}")
            print(f"Original labels shape: {ground_truth_labels.shape}")
        if self.segment_len is None:
            # Return full length without slicing or padding
            segment = spec.copy()
            segment_labels = ground_truth_labels.copy()
            print(f"DEBUG: Returning full segment for {file_path}, shape: {segment.shape}")
        else:
            if T < self.segment_len:
                # pad both spectrogram and labels
                if self.verbose:
                    print(f"Padding short segment from length {T} to {self.segment_len}")
                
                padded_spec = np.zeros((F, self.segment_len))
                padded_spec[:, :T] = spec
                
                padded_labels = np.zeros(self.segment_len, dtype=ground_truth_labels.dtype)
                padded_labels[:T] = ground_truth_labels
                
                spec = padded_spec
                ground_truth_labels = padded_labels
                T = self.segment_len
                print(f"DEBUG: Padded segment for {file_path}, new shape: {spec.shape}")

            # Directly slice the memmapped arrays
            if self.infinite_dataset:
                # Use random start for infinite dataset
                start = random.randint(0, T - self.segment_len)
            else:
                # Use deterministic start for finite dataset based on index
                # This ensures we get consistent segments for the same file
                start = 0  # Always start from the beginning for consistency
            
            segment = spec[:, start:start+self.segment_len].copy()  # Copy forces load into memory
            segment_labels = ground_truth_labels[start:start+self.segment_len].copy()
            print(f"DEBUG: Sliced segment for {file_path}, from position {start}, shape: {segment.shape}")

        # global z-score normalization
        mean_val = np.mean(segment)
        std_val = np.std(segment)
        segment = (segment - mean_val) / (std_val + 1e-8)

        segment = torch.from_numpy(segment).float()
        segment_labels = torch.from_numpy(segment_labels).long()
        
        print(f"DEBUG: Returning segment from {file_path}")        
        return segment, segment_labels, os.path.basename(file_path)

@timed_operation("collate_fn")
def collate_fn(batch, segment_length=500, mask_p=0.75, verbose=False):

    # Simplified collate function without timing blocks
    specs, labels, filenames = zip(*batch)
    
    # Stack tensors
    segs = torch.stack(specs, dim=0)
    labels = torch.stack(labels, dim=0)
    
    # Create copies for masking
    full_spectrogram = segs.clone()
    context_spectrogram = segs.clone()
    target_spectrogram = segs.clone()
    
    # Get batch size and dimensions
    B, F, T = segs.shape
    
    # Create the mask
    mask = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        remaining_timesteps_to_mask = int(mask_p * T)
        while remaining_timesteps_to_mask > 0:
            block_size = min(random.randint(1, 10), remaining_timesteps_to_mask)
            t_start = random.randint(0, T - block_size)
            mask[b, t_start:t_start+block_size] = True
            remaining_timesteps_to_mask -= block_size
    
    # Apply the mask
    for b in range(B):
        context_spectrogram[b, :, mask[b]] = 0  # Zero out masked regions in context
        target_spectrogram[b, :, ~mask[b]] = 0  # Zero out unmasked regions in target
    
    if verbose:
        print(f"Batch shapes - Input: {segs.shape}, Mask: {mask.shape}")
    
    return full_spectrogram, target_spectrogram, context_spectrogram, labels, mask, filenames
