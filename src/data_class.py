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

            # Replace min-max normalization with z-score normalization
            mean_val = np.mean(segment, axis=1, keepdims=True)   # (D,1)
            std_val = np.std(segment, axis=1, keepdims=True)     # (D,1)
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
    # batch: list of tuples (spectrogram, labels)
    # Unzip the batch into separate lists
    specs, labels = zip(*batch)
    
    # stack -> (B,F,T)
    segs = torch.stack(specs, dim=0)  # (B,F,T)
    labels = torch.stack(labels, dim=0)  # (B,T)
    B, F, T = segs.shape
    if verbose:
        print(f"Stacked batch shape: {segs.shape}")

    # Create mask along time dimension (T)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        remaining_timesteps_to_mask = int(mask_p * T)
        while remaining_timesteps_to_mask > 0:
            block_size = random.randint(1, remaining_timesteps_to_mask)
            t_start = random.randint(0, T - block_size)
            mask[b, t_start:t_start+block_size] = True
            remaining_timesteps_to_mask -= block_size

    # Clone spectrograms
    full_spectrogram = segs.clone()       # (B,F,T)
    context_spectrogram = segs.clone()    # (B,F,T)
    target_spectrogram = segs.clone()     # (B,F,T)

    # Apply masking using noise
    for b in range(B):
        noise_context = torch.randn_like(context_spectrogram[b, :, mask[b]])
        noise_target = torch.randn_like(target_spectrogram[b, :, ~mask[b]])
        
        context_spectrogram[b, :, mask[b]] = noise_context
        target_spectrogram[b, :, ~mask[b]] = noise_target

    # Return mask tensor directly
    file_names = [f"dummy_{i}.npy" for i in range(B)]
    return full_spectrogram, target_spectrogram, context_spectrogram, labels, mask, file_names
