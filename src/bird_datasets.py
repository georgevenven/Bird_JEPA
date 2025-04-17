import os, random, numpy as np, torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from typing import List, Tuple, Dict

# ---------- helpers ----------
def load_np(path: str):
    # mmap read; returns dict‑like object
    return np.load(path, allow_pickle=True, mmap_mode='r')

# ---------- dataset ----------
class BirdSpectrogramDataset(Dataset):
    """
    Loads a directory of `.npz` spectrogram files.
    Each file must contain keys 's' and 'labels'.
    """
    def __init__(self,
                 data_dir: str,
                 segment_len: int = 50,
                 infinite: bool = True,
                 verbose: bool = False):
        self.paths: List[str] = [e.path for e in os.scandir(data_dir)]
        if not self.paths:
            raise RuntimeError(f'no files found in {data_dir}')
        self.segment_len = segment_len
        self.infinite = infinite
        self.verbose = verbose

    # ---------------------------------------------------

    def __len__(self):
        return int(1e5) if self.infinite else len(self.paths)

    # ---------------------------------------------------

    def _pull_segment(self, spec: np.ndarray,
                      labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pads or slices (F,T) spec to fixed T = segment_len and
        returns (segment, segment_labels).  Zero‑pads if short.
        """
        F, T = spec.shape
        if self.segment_len is None:         # "use full context"
            return spec.copy(), labels.copy()

        if T < self.segment_len:             # pad
            pad_spec = np.zeros((F, self.segment_len), spec.dtype)
            pad_lab  = np.zeros((self.segment_len,), labels.dtype)
            pad_spec[:, :T], pad_lab[:T] = spec, labels
            return pad_spec, pad_lab

        # T ≥ segment_len: choose start index
        start = random.randint(0, T - self.segment_len)
        end = start + self.segment_len
        return spec[:, start:end], labels[start:end]

    # ---------------------------------------------------

    def __getitem__(self, idx: int):
        if self.infinite:
            idx = random.randint(0, len(self.paths) - 1)
        path = self.paths[idx]
        npz  = load_np(path)
        spec = npz['s'][:, :-1]              # drop final STFT frame
        labels = npz['labels']

        # --- align label length to spectrogram length -----------------
        T = spec.shape[1]
        if labels.shape[0] > T:           # too long → truncate
            labels = labels[:T]
        elif labels.shape[0] < T:         # too short → zero‑pad
            pad = np.zeros(T, dtype=labels.dtype)
            pad[: labels.shape[0]] = labels
            labels = pad

        seg, seg_lab = self._pull_segment(spec, labels)

        # global z‑score
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)

        return (torch.from_numpy(seg).float(),
                torch.from_numpy(seg_lab).long(),
                os.path.basename(path))
