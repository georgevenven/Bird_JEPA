import numpy as np
import matplotlib.pyplot as plt

# Load the spectrogram data from npz file
data = np.load('/home/george-vengrovski/Downloads/XC173760_segment_0.npz')
spectrogram = data['s']  # Access the 's' array

print(spectrogram.shape)

# Crop the spectrogram as specified
spectrogram = spectrogram[10:216, :]

# Create the figure and plot without colorbar
plt.figure(figsize=(30, 6))
plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
plt.xlabel('Time Frame')
plt.ylabel('Frequency Bin')
plt.title('Spectrogram')

plt.show()