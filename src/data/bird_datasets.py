import os, random, numpy as np, torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from typing import List, Tuple, Dict
import pandas as pd
from utils import build_label_map
import zipfile

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
                 verbose: bool = False,
                 csv_path: str | None = None):
        self.paths: List[str] = [
            e.path for e in os.scandir(data_dir)
            if e.is_file() and (e.name.lower().endswith(".npz") or e.name.lower().endswith(".pt"))
        ]
        if not self.paths:
            # Gracefully handle empty directory: create empty CSV with just column labels
            empty_csv_path = os.path.join(data_dir, "empty.csv")
            if not os.path.exists(empty_csv_path):
                pd.DataFrame(columns=["filename", "primary_label"]).to_csv(empty_csv_path, index=False)
            self.paths = []  # keep dataset empty, but do not crash
        self.segment_len = segment_len
        self.infinite = infinite
        self.verbose = verbose

        if csv_path is not None:
            self.fname2lab, self.classes = build_label_map(csv_path)
            self.label_to_idx = {c:i for i,c in enumerate(self.classes)}
        else:
            self.fname2lab, self.classes, self.label_to_idx = {}, [], {}

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
        if not self.paths:
            raise IndexError("No data available in dataset.")
        if self.infinite:
            idx = random.randint(0, len(self.paths) - 1)
        path = self.paths[idx]
        path_obj = Path(path)
        if path_obj.suffix == ".pt":
            obj = torch.load(path, mmap=True, map_location="cpu", weights_only=True)

            # ----- unwrap ----------------------------------------------------
            if isinstance(obj, dict):          # new .pt format {"s": tensor}
                spec_t   = obj["s"]
                labels_t = obj.get("labels")    # may be None
            else:                              # legacy raw tensor
                spec_t, labels_t = obj, None

            # ----- always hand back numpy ------------------------------------
            if isinstance(spec_t, torch.Tensor):
                spec = spec_t.half().cpu().numpy()
            else:
                spec = spec_t.astype(np.float32)

            if labels_t is None:
                labels = np.zeros(spec.shape[1], dtype=np.int32)
            elif isinstance(labels_t, torch.Tensor):
                labels = labels_t.cpu().numpy()
            else:
                labels = labels_t

            # ── slice random segment for training ─────────────────────────
            if self.segment_len is not None:                 # training mode
                spec, labels = self._pull_segment(spec, labels)

            fname = Path(path).name
            return spec, labels, fname
        else:  # .npz
            try:
                npz = load_np(path)
            except (zipfile.BadZipFile, ValueError, OSError) as exc:
                if getattr(self, 'verbose', False):
                    print(f"[dataset] skip {path}: {exc}")
                if self.infinite:
                    return self.__getitem__(random.randint(0, len(self.paths) - 1))
                else:
                    raise IndexError("corrupt file skipped")
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

        fname = Path(path).name        # robust whether path is str or Path
        return (torch.from_numpy(seg).float(),
                torch.from_numpy(seg_lab).long(),
                fname)

    def label_idx(self, npz_name: str):
        """
        npz_name: 'XC12345.npz' → returns int label or -1 if unknown
        """
        key = Path(npz_name).with_suffix('.ogg').name
        return self.label_to_idx.get(self.fname2lab.get(key, ''), -1)

class TorchSpecDataset(BirdSpectrogramDataset):
    def _load(self, path):
        return torch.load(path, mmap=True)["s"]
