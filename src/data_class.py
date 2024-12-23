# data_class.py
import torch
from torch.utils.data import Dataset
import random
import os
import numpy as np

class BirdJEPA_Dataset(Dataset):
    def __init__(self, data_dir, segment_len=50, verbose=False):
        # we assume each file has spec shape: (D,T)
        # we want to extract a contiguous segment of length segment_len along time dimension T
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        self.segment_len = segment_len
        self.verbose = verbose
        self.epoch_size = 10000  # number of samples per epoch

        if self.verbose:
            print(f"Initialized dataset with {len(self.files)} files.")

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        # pick a random file
        fpath = random.choice(self.files)
        if self.verbose:
            print(f"Selected file: {fpath}")

        try:
            data = np.load(fpath)
            spec = data['s']  # we assume shape: (F,T)
            ground_truth_labels = data['labels']  # shape: T
            

            # F stands for frequency bins, T stands for time frames
            F, T = spec.shape

            if self.verbose:
                print(f"Initial spec shape: {spec.shape}")
                print(f"Initial ground_truth_labels shape: {ground_truth_labels.shape}")

            if T < self.segment_len:
                # Pad both spectrogram and labels
                if self.verbose:
                    print(f"Padding short segment from length {T} to {self.segment_len}")
                
                # Create padded spectrogram with zeros
                padded_spec = np.zeros((F, self.segment_len))
                padded_spec[:, :T] = spec  # Copy original data
                
                # Create padded labels with zeros (or another padding value if needed)
                padded_labels = np.zeros(self.segment_len, dtype=ground_truth_labels.dtype)
                padded_labels[:T] = ground_truth_labels  # Copy original labels
                
                spec = padded_spec
                ground_truth_labels = padded_labels
                T = self.segment_len  # Update T to new length

            # select a time slice of length segment_len
            start = random.randint(0, T - self.segment_len)
            if self.verbose:
                print(f"Selected start index: {start}")

            # extract segment and corresponding labels
            segment = spec[:, start:start+self.segment_len]
            segment_labels = ground_truth_labels[start:start+self.segment_len]  # Get corresponding labels

            # Replace column-wise z-score normalization with global z-score normalization
            mean_val = np.mean(segment)    # Single scalar mean
            std_val = np.std(segment)      # Single scalar std
            segment = (segment - mean_val) / (std_val + 1e-8)    # Add epsilon for numerical stability

            segment = torch.from_numpy(segment).float()  # shape (D, segment_len)
            segment_labels = torch.from_numpy(segment_labels).long()  # Convert labels to tensor
            
            if self.verbose:
                print(f"Returning segment with shape: {segment.shape} (D,T)")
                print(f"Returning labels with shape: {segment_labels.shape} (T)")
            return segment, segment_labels
        except Exception as e:
            if self.verbose:
                print(f"Error loading file {fpath}: {e}, trying another file.")
            return self.__getitem__(random.randint(0, len(self.files)-1))

def collate_fn(batch, segment_length=500, mask_p=0.75, verbose=False):
    # Unzip the batch into separate lists
    specs, labels = zip(*batch)
    
    # stack -> (B,F,T)
    segs = torch.stack(specs, dim=0)  # (B,F,T)
    labels = torch.stack(labels, dim=0)  # (B,T)
    
    # Create full_spectrogram before any masking
    full_spectrogram = segs.clone()  # This should be completely unmasked
    
    # Get batch size and sequence length
    B, F, T = segs.shape  # Define B here before using it
    
    if verbose:
        print(f"DEBUG: full_spectrogram shape before any processing: {full_spectrogram.shape}")
        print(f"DEBUG: full_spectrogram contains -1?: {(full_spectrogram == -1).any().item()}")
    
    # Create mask
    mask = torch.zeros(B, T, dtype=torch.bool)  # Now B is defined
    
    # Create mask along time dimension (T)
    for b in range(B):
        remaining_timesteps_to_mask = int(mask_p * T)
        while remaining_timesteps_to_mask > 0:
            block_size = random.randint(1, remaining_timesteps_to_mask)
            t_start = random.randint(0, T - block_size)
            mask[b, t_start:t_start+block_size] = True
            remaining_timesteps_to_mask -= block_size

    # Clone spectrograms
    context_spectrogram = segs.clone()    # (B,F,T)
    target_spectrogram = segs.clone()     # (B,F,T)

    # Apply masking using zeros instead of noise
    for b in range(B):
        context_spectrogram[b, :, mask[b]] = 0  # Replace with zeros for masked regions
        target_spectrogram[b, :, ~mask[b]] = 0  # Zero out unmasked regions in target

    # Return mask tensor directly
    file_names = [f"dummy_{i}.npy" for i in range(B)]
    
    # Add shape and value range debugging
    if verbose:
        print("\nShape Information:")
        print(f"full_spectrogram: {full_spectrogram.shape}")
        print(f"target_spectrogram: {target_spectrogram.shape}")
        print(f"context_spectrogram: {context_spectrogram.shape}")
        print(f"labels: {labels.shape}")
        print(f"mask: {mask.shape}")
        
        print("\nValue Ranges:")
        print(f"full_spectrogram: min={full_spectrogram.min().item():.3f}, max={full_spectrogram.max().item():.3f}, avg={full_spectrogram.mean().item():.3f}")
        print(f"target_spectrogram: min={target_spectrogram.min().item():.3f}, max={target_spectrogram.max().item():.3f}, avg={target_spectrogram.mean().item():.3f}")
        print(f"context_spectrogram: min={context_spectrogram.min().item():.3f}, max={context_spectrogram.max().item():.3f}, avg={context_spectrogram.mean().item():.3f}")
        
    return full_spectrogram, target_spectrogram, context_spectrogram, labels, mask, file_names
