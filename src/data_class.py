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
    def __init__(self, data_dir, segment_len=50, verbose=False):
        self.data_dir = data_dir
        self.segment_len = segment_len
        self.verbose = verbose
        
        # preload file paths into memory for fast access
        self.file_paths = [entry.path for entry in os.scandir(data_dir)]
        self.file_count = len(self.file_paths)
        
        if self.verbose:
            print(f"Initialized dataset with {self.file_count} files.")
                
    def __len__(self):
        # return a very large number to simulate an infinite dataset
        return int(1e5)
    
    def _get_random_file(self):
        """Get a random file path from the preloaded file paths."""
        idx = random.randint(0, self.file_count - 1)
        return self.file_paths[idx]
    
    def _load_file(self, file_path):
        """Load file with memory mapping for efficient partial reads."""
        return np.load(file_path, allow_pickle=True, mmap_mode='r')

    @timed_operation("data_loading")
    def __getitem__(self, idx):
        # ignore the provided idx and select a random file
        file_path = self._get_random_file()
        
        # Load with memory mapping
        data = self._load_file(file_path)
        spec = data['s']  # Memory-mapped array
        ground_truth_labels = data['labels'] # Memory-mapped array

        F, T = spec.shape
        
        if self.verbose:
            print(f"Loaded file: {file_path}")
            print(f"Original spectrogram shape: {spec.shape}")
            print(f"Original labels shape: {ground_truth_labels.shape}")

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

            # Directly slice the memmapped arrays
            start = random.randint(0, T - self.segment_len)
            segment = spec[:, start:start+self.segment_len].copy()  # Copy forces load into memory
            segment_labels = ground_truth_labels[start:start+self.segment_len].copy()
        else:
            # Directly slice the memmapped arrays
            start = random.randint(0, T - self.segment_len)
            segment = spec[:, start:start+self.segment_len].copy()  # Copy forces load into memory
            segment_labels = ground_truth_labels[start:start+self.segment_len].copy()

        # global z-score normalization
        mean_val = np.mean(segment)
        std_val = np.std(segment)
        segment = (segment - mean_val) / (std_val + 1e-8)

        segment = torch.from_numpy(segment).float()
        segment_labels = torch.from_numpy(segment_labels).long()
                
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

class BirdCLEFDataset(Dataset):
    def __init__(self, root_dir, transform=None, debug=False):
        with Timer("clef_dataset_initialization", debug=debug):
            self.root_dir = Path(root_dir)
            self.transform = transform
            self.debug = debug
            self.files = []
            for file_path in self.root_dir.rglob('*.mp3'):
                self.files.append(file_path)
            if self.debug:
                print(f"[Dataset] Found {len(self.files)} files")
    def __len__(self):
        return len(self.files)
    @timed_operation("clef_dataset_getitem")
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        with Timer("audio_loading", debug=self.debug):
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if self.transform:
                waveform = self.transform(waveform)
            spectrogram = torchaudio.transforms.MelSpectrogram()(waveform)
            spectrogram = torch.log(spectrogram + 1e-9)
            spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-9)
            if self.debug and idx % 100 == 0:
                print(f"[Dataset] Loaded item {idx}")
            return spectrogram

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class BirdCLEFDataModule:
    def __init__(self, data_dir, batch_size=32, num_workers=4, debug=False):
        with Timer("data_module_initialization", debug=debug):
            self.data_dir = Path(data_dir)
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.debug = debug
            self.train_dataset = BirdCLEFDataset(
                self.data_dir / 'train',
                debug=debug
            )
            self.val_dataset = BirdCLEFDataset(
                self.data_dir / 'val',
                debug=debug
            )
            if self.debug:
                print(f"[DataModule] Train samples: {len(self.train_dataset)}")
                print(f"[DataModule] Val samples: {len(self.val_dataset)}")
    @timed_operation("data_module_setup")
    def setup(self):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        if self.debug:
            print(f"[DataModule] Train batches: {len(self.train_loader)}")
            print(f"[DataModule] Val batches: {len(self.val_loader)}")
    def get_train_loader(self):
        return self.train_loader
    def get_val_loader(self):
        return self.val_loader
