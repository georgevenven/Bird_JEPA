import pandas as pd, re
from pathlib import Path
from torch.utils.data import DataLoader, SequentialSampler
from bird_datasets import BirdSpectrogramDataset

class BirdDataModule:
    """
    * parses the train CSV once
    * exposes .train_loader / .val_loader
    * exposes .label_to_idx, .idx_to_label, .num_classes
    """
    def __init__(self,
                 train_csv: str,
                 train_dir: str,
                 val_dir: str,
                 context_len: int,
                 batch_size: int,
                 infinite_train: bool = True,
                 num_workers: int = 0):

        df = pd.read_csv(train_csv)
        self.classes = sorted(df['primary_label'].unique())
        self.label_to_idx = {c:i for i,c in enumerate(self.classes)}

        # quick lookup by XC / iNat / CSA id
        self._id_rx = re.compile(r'(XC\d+|iNat\d+|CSA\d+)')
        self._map   = {}
        for _, r in df.iterrows():
            fname = Path(r['filename' if 'filename' in r else 'file_name']).name
            m = self._id_rx.search(fname)
            if m:
                self._map[m.group(1)] = r['primary_label']

        self.val_ds   = BirdSpectrogramDataset(val_dir,
                                               segment_len=context_len,
                                               infinite=False)

        if train_dir and Path(train_dir).is_dir():
            self.train_ds = BirdSpectrogramDataset(train_dir,
                                                   segment_len=context_len,
                                                   infinite=infinite_train)
            self.train_loader = DataLoader(
                self.train_ds,
                batch_size=batch_size,
                sampler=None,              # shuffle inside dataset via infinite=True
                shuffle=False,
                collate_fn=self.stack_collate,
                num_workers=num_workers,
                pin_memory=True)
        else:
            self.train_ds = self.train_loader = None

        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=1,
            sampler=None,
            shuffle=False,
            collate_fn=self.stack_collate,
            num_workers=0)

    # ------------------------------------------------------------------
    def label_idx(self, filename: str):
        m = self._id_rx.search(filename)
        if not m: return -1
        return self.label_to_idx.get(self._map.get(m.group(1), ''), -1)

    @property
    def num_classes(self): return len(self.classes)

    # ------------------------------------------------------------------
    # plain "stack only" collate – NO masking
    def stack_collate(self, batch):
        import torch
        specs, labels, fnames = zip(*batch)
        return torch.stack(specs), torch.stack(labels), fnames
