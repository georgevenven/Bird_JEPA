# test_overfit_single_batch.py
import unittest
import torch
import os
import shutil
import numpy as np
from torch.utils.data import DataLoader
import sys
import torch.nn.functional as F

# adjust the path as needed to find your model and data_class modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model import BirdJEPA
from data_class import BirdJEPA_Dataset, collate_fn
import torch.optim as optim

class TestOverfitSingleBatch(unittest.TestCase):
    def setUp(self):
        # create a temporary directory with dummy data
        self.temp_dir = "temp_data_overfit"
        os.makedirs(self.temp_dir, exist_ok=True)
        # create some dummy npz files
        for i in range(2):
            # s: T x D, let's say T=10, D=200
            s = np.random.rand(10, 200)
            labels = np.random.randint(0, 10, size=(10,))
            np.savez(os.path.join(self.temp_dir, f"sample_{i}.npz"), s=s, labels=labels)

        # Create a custom dataset class for testing that returns a finite length
        class TestDataset(BirdJEPA_Dataset):
            def __len__(self):
                return len(self.files)  # Return actual number of files instead of 1e12

        self.dataset = TestDataset(data_dir=self.temp_dir, segment_len=10, verbose=False)
        self.dl = DataLoader(
            self.dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, segment_length=10, mask_p=0.5, verbose=False)
        )
        
        # get one batch
        batch = next(iter(self.dl))
        full_spectrogram, target_spectrogram, context_spectrogram, ground_truth_labels, vocalization, file_names = batch
        B, T, D = context_spectrogram.shape
        self.input_dim = D

        self.model = BirdJEPA(input_dim=self.input_dim, hidden_dim=32)
        self.opt = optim.Adam(list(self.model.context_encoder.parameters()) + list(self.model.predictor.parameters()) + list(self.model.decoder.parameters()), lr=1e-3)

        self.context_spectrogram = context_spectrogram
        self.target_spectrogram = target_spectrogram

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_overfit_single_batch(self):
        initial_loss = None
        for step in range(20):
            loss = self.model.training_step(self.context_spectrogram, self.target_spectrogram)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.model.update_ema()

            print(f"Step {step}, Loss: {loss.item():.4f}")

            if step == 0:
                initial_loss = loss.item()
            if step == 19:
                final_loss = loss.item()
                print(f"\nInitial loss: {initial_loss:.4f}")
                print(f"Final loss: {final_loss:.4f}")
                self.assertTrue(final_loss < initial_loss, 
                              f"final loss {final_loss} is not less than initial loss {initial_loss}")

    def test_decoder(self):
        # Get representations from the encoder
        context_repr, _ = self.model.context_encoder(self.context_spectrogram)
        
        # Pass through decoder
        reconstruction = self.model.decoder(context_repr)
        
        # Check output shape matches input
        self.assertEqual(reconstruction.shape, self.context_spectrogram.shape)
        
        # Check reconstruction quality
        reconstruction_loss = F.mse_loss(reconstruction, self.context_spectrogram)
        self.assertLess(reconstruction_loss, 1.0)  # adjust threshold as needed

if __name__ == '__main__':
    unittest.main(verbosity=2)
