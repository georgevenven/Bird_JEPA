# test_model.py
import unittest
import torch
import os
import sys
import math

# adjust path as needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model import BirdJEPA

class TestBirdJEPA(unittest.TestCase):
    def setUp(self):
        self.hidden_dim = 64
        self.input_dim = 64  # freq bins dimension
        self.model = BirdJEPA(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        self.model.eval()

        B, T, D = 2, 50, self.input_dim
        full_spectrogram = torch.randn(B, T, D)

        # create a mask
        mask = torch.zeros(B, D, dtype=torch.bool)
        for b in range(B):
            masked_features = int(0.5 * D)
            idxs = torch.randperm(D)[:masked_features]
            mask[b, idxs] = True

        context_spectrogram = full_spectrogram.clone()
        target_spectrogram = torch.zeros_like(full_spectrogram)

        # apply mask
        for b in range(B):
            for t in range(T):
                context_spectrogram[b, t, mask[b]] = -1.0
                target_spectrogram[b, t, mask[b]] = full_spectrogram[b, t, mask[b]]

        self.context_spectrogram = context_spectrogram
        self.target_spectrogram = target_spectrogram

    def test_forward(self):
        with torch.no_grad():
            pred, target = self.model(self.context_spectrogram, self.target_spectrogram)
        self.assertEqual(pred.shape, target.shape)
        self.assertEqual(pred.size(0), self.context_spectrogram.size(0))
        self.assertEqual(pred.size(1), self.context_spectrogram.size(1))
        self.assertEqual(pred.size(2), self.hidden_dim)

    def test_training_step(self):
        loss = self.model.training_step(self.context_spectrogram, self.target_spectrogram)
        self.assertTrue(loss.item() >= 0)

if __name__ == '__main__':
    unittest.main()
