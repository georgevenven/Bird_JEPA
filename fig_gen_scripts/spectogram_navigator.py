import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from matplotlib.widgets import Button
from matplotlib.backend_bases import KeyEvent
import torch

# ===== MODIFY THESE PARAMETERS =====
NPZ_DIR = '/home/george-vengrovski/Documents/projects/Bird_JEPA/BirdCLEF/finetune_val'
PATTERN = '*.pt'        # File pattern to match
SPEC_KEY = 's'           # Key for the spectrogram in the NPZ file
CROP_TOP = 0            # Crop from the top
CROP_BOTTOM = 128        # Crop to the bottom
# ==================================

class SpectogramNavigator:
    def __init__(self, npz_dir, pattern='*.npz', spectrogram_key='s'):
        self.npz_files = sorted(glob.glob(os.path.join(npz_dir, pattern)))
        
        if not self.npz_files:
            raise ValueError(f"No NPZ or PT files found in {npz_dir} with pattern {pattern}")
        
        self.current_index = 0
        self.spectrogram_key = spectrogram_key
        self.fig = None
        self.ax = None
        self.img = None
        self.crop_top = 0
        self.crop_bottom = 100000
        
    def load_current_spectrogram(self):
        """Load the current spectrogram from the list"""
        current_file = self.npz_files[self.current_index]
        try:
            if current_file.endswith('.npz'):
                data = np.load(current_file)
                spectrogram = data[self.spectrogram_key]
            elif current_file.endswith('.pt'):
                data = torch.load(current_file)
                spectrogram = data[self.spectrogram_key]
            else:
                raise ValueError(f"Unsupported file type: {current_file}")
            # Z-score normalization per spectrogram
            spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)
            # Crop the spectrogram
            if self.crop_bottom > 0 and self.crop_bottom > self.crop_top:
                spectrogram = spectrogram[self.crop_top:self.crop_bottom, :]
            return spectrogram, os.path.basename(current_file)
        except Exception as e:
            print(f"Error loading {current_file}: {e}")
            return np.zeros((10, 10)), f"Error: {current_file}"
    
    def on_key_press(self, event):
        """Handle key press events for navigation"""
        if event.key == 'right':
            self.current_index = (self.current_index + 1) % len(self.npz_files)
            self.update_plot()
        elif event.key == 'left':
            self.current_index = (self.current_index - 1) % len(self.npz_files)
            self.update_plot()
    
    def update_plot(self):
        """Update the plot with the current spectrogram"""
        spectrogram, title = self.load_current_spectrogram()
        
        if self.img is None:
            self.img = self.ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
        else:
            self.img.set_data(spectrogram)
            self.img.set_extent((0, spectrogram.shape[1], 0, spectrogram.shape[0]))
        
        self.ax.set_title(f"[{self.current_index+1}/{len(self.npz_files)}] {title}")
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Display the spectogram navigator"""
        self.fig, self.ax = plt.subplots(figsize=(30, 6))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Add navigation instructions
        self.fig.text(0.5, 0.01, 
                 "Navigation: Left/Right Arrow Keys to change files", 
                 ha='center', fontsize=12)
        
        # Initialize with the first spectrogram
        self.update_plot()
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.show()

if __name__ == '__main__':
    navigator = SpectogramNavigator(NPZ_DIR, PATTERN, SPEC_KEY)
    navigator.crop_top = CROP_TOP
    navigator.crop_bottom = CROP_BOTTOM
    navigator.show() 