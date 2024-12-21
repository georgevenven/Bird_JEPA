import unittest
import torch
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# adjust path as needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_class import BirdJEPA_Dataset, collate_fn
from torch.utils.data import DataLoader

class TestDataClass(unittest.TestCase):
    def setUp(self):
        # set up dataset and dataloader
        self.data_dir = "/media/george-vengrovski/George-SSD/llb_stuff/llb3_test"
        self.dataset = BirdJEPA_Dataset(data_dir=self.data_dir, segment_len=1000, verbose=True)
        self.loader = DataLoader(self.dataset, batch_size=2,
                               collate_fn=lambda batch: collate_fn(
                                   batch,
                                   segment_length=1000,
                                   mask_p=0.5,
                                   verbose=True
                               ))

    def test_dataclass(self):
        # get a single batch
        batch = next(iter(self.loader))
        full_spectrogram, target_spectrogram, context_spectrogram, ground_truth_labels, mask, file_names = batch

        print("==== test_dataclass ====")
        print("full_spectrogram shape:", full_spectrogram.shape)
        print("target_spectrogram shape:", target_spectrogram.shape)
        print("context_spectrogram shape:", context_spectrogram.shape)
        print("ground_truth_labels shape:", ground_truth_labels.shape)
        print("mask shape:", mask.shape)
        print("file_names:", file_names)

        os.makedirs("test_outputs", exist_ok=True)

        # Plot corresponding labels in bottom row
        labels = ground_truth_labels[0].cpu().numpy()
        
        # Create custom colormap with black for 0
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        colors[0] = [0, 0, 0, 1]  # Set 0 label to black
        custom_cmap = plt.matplotlib.colors.ListedColormap(colors)
        
        # Adjust the subplot heights to make label plots shorter
        fig, axs = plt.subplots(2, 3, figsize=(15, 8), 
                               gridspec_kw={'height_ratios': [3, 0.375]})  # Reduced bottom row height
        
        # Plot spectrograms in top row
        axs[0,0].imshow(full_spectrogram[0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
        axs[0,0].set_title('Full Spectrogram')
        axs[0,0].set_xlabel('Time frames')
        axs[0,0].set_ylabel('Frequency bins')

        axs[0,1].imshow(context_spectrogram[0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
        axs[0,1].set_title('Context Spectrogram (masked)')
        axs[0,1].set_xlabel('Time frames')

        axs[0,2].imshow(target_spectrogram[0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
        axs[0,2].set_title('Target Spectrogram (masked parts)')
        axs[0,2].set_xlabel('Time frames')

        # Update label plots with new colormap
        axs[1,0].imshow(labels.reshape(1, -1), aspect='auto', cmap=custom_cmap, vmin=0, vmax=max(9, labels.max()))
        axs[1,0].set_title('Labels')
        axs[1,0].set_xlabel('Time frames')
        axs[1,0].set_yticks([])
        
        axs[1,1].imshow(labels.reshape(1, -1), aspect='auto', cmap=custom_cmap, vmin=0, vmax=max(9, labels.max()))
        axs[1,1].set_xlabel('Time frames')
        axs[1,1].set_yticks([])
        
        axs[1,2].imshow(labels.reshape(1, -1), aspect='auto', cmap=custom_cmap, vmin=0, vmax=max(9, labels.max()))
        axs[1,2].set_xlabel('Time frames')
        axs[1,2].set_yticks([])

        # Add colorbar without legend
        cbar = plt.colorbar(axs[1,2].images[0], ax=axs[1,:].ravel().tolist(), ticks=np.arange(0, max(9, labels.max()) + 1))
        cbar.set_label('Label Values')

        plt.tight_layout()
        plt.savefig(os.path.join("test_outputs", "test_dataclass_masking.png"))
        plt.close()

        # simple assertions
        self.assertIsInstance(full_spectrogram, torch.Tensor)
        self.assertIsInstance(context_spectrogram, torch.Tensor)
        self.assertIsInstance(target_spectrogram, torch.Tensor)
        self.assertIsInstance(ground_truth_labels, torch.Tensor)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertTrue(len(file_names) == full_spectrogram.size(0))
        # check masking occurred by verifying noise in context spectrogram where mask is True
        self.assertTrue((context_spectrogram[0, :, mask[0]] != full_spectrogram[0, :, mask[0]]).all())

if __name__ == '__main__':
    unittest.main()
