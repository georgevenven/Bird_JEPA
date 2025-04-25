import numpy as np
import matplotlib.pyplot as plt
import os
import torch

# Load the spectrogram data from npz or pt file
def load_spectrogram_and_labels(path):
    ext = os.path.splitext(path)[1]
    if ext == '.npz':
        data = np.load(path)
        spectrogram = data['s']
        labels = data['labels'] if 'labels' in data else None
    elif ext == '.pt':
        data = torch.load(path, map_location='cpu')
        spectrogram = data['s'].numpy() if isinstance(data['s'], torch.Tensor) else data['s']
        labels = data['labels'].numpy() if 'labels' in data and isinstance(data['labels'], torch.Tensor) else data.get('labels', None)
    else:
        raise ValueError('Unsupported file type: ' + ext)
    return spectrogram, labels

# Specify the path to your file (npz or pt)
file_path = '/media/george-vengrovski/Desk SSD/BirdJEPA/llb3_specs/llb3_2164_2018_04_28_11_20_08.pt'

spectrogram, labels = load_spectrogram_and_labels(file_path)

print(spectrogram.shape)

# Crop the spectrogram as specified
spectrogram = spectrogram[10:216, :]
if labels is not None:
    labels = labels[slice(0, spectrogram.shape[1])]

# Create the figure and plot without colorbar
fig, ax = plt.subplots(2, 1, figsize=(30, 7), gridspec_kw={'height_ratios': [12, 1]})

ax[0].imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
ax[0].set_xlabel('Time Frame')
ax[0].set_ylabel('Frequency Bin')
ax[0].set_title('Spectrogram')

if labels is not None:
    ax[1].imshow(labels[np.newaxis, :], aspect='auto', cmap='tab20', interpolation='nearest')
    ax[1].set_yticks([])
    ax[1].set_xlabel('Time Frame')
    ax[1].set_title('Labels')
else:
    ax[1].axis('off')

plt.tight_layout()
plt.show()