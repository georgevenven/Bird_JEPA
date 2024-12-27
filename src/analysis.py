# analysis.py
import os
import sys
import argparse
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import shutil
import random
import glob
from data_class import BirdJEPA_Dataset, collate_fn
from torch.utils.data import DataLoader
from utils import load_model
import glasbey
import torch.nn.functional as F

def generate_hdbscan_labels(array, min_samples=1, min_cluster_size=5000):
    """
    Generate labels for data points using the HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) clustering algorithm.

    Parameters:
    - array: ndarray of shape (n_samples, n_features)
      The input data to cluster.

    - min_samples: int, default=5
      The number of samples in a neighborhood for a point to be considered as a core point.

    - min_cluster_size: int, default=5
      The minimum number of points required to form a cluster.

    Returns:
    - labels: ndarray of shape (n_samples)
      Cluster labels for each point in the dataset. Noisy samples are given the label -1.
    """

    import hdbscan

    # Create an HDBSCAN object with the specified parameters.
    hdbscan_model = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)

    # Fit the model to the data and extract the labels.
    labels = hdbscan_model.fit_predict(array)

    print(f"discovered labels {np.unique(labels)}")

    return labels

def load_data(data_dir, context=1000, batch_size=1, shuffle=True):
    """
    Load data using the BirdJEPA_Dataset class
    
    Parameters:
    - data_dir: directory containing the data files
    - context: context window size (used as segment_len)
    - batch_size: batch size for the DataLoader
    - shuffle: whether to shuffle the data
    
    Returns:
    - DataLoader object that yields (data, ground_truth_label, vocalization, file_path)
    """
    dataset = BirdJEPA_Dataset(data_dir, segment_len=context)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, segment_length=context)
    )
    
    # Wrap the loader to only return the values we need
    class DataLoaderWrapper:
        def __init__(self, loader):
            self.loader = loader
        
        def __iter__(self):
            for full_spec, target_spec, context_spec, labels, vocal, files in self.loader:
                # Return only what plot_umap_projection expects
                yield full_spec, labels, vocal, files
    
    return DataLoaderWrapper(loader)

def plot_umap_projection(model, device, data_dirs, category_colors_file="test_llb16", samples=1e6, file_path='category_colors.pkl',
                         layer_index=None, dict_key=None, 
                         context=1000, save_name=None, raw_spectogram=False, save_dict_for_analysis=True, 
                         remove_non_vocalization=False, min_cluster_size=500):
    """
    Parameters:
    - data_dirs: list of data directories to analyze
    note: we assume data in data_dirs are spectrogram npz or similar processed by load_data()
    model: BirdJEPA model loaded from utils.py
    """
    print("\nStarting UMAP projection...")
    
    # Initialize storage arrays
    spec_arr = []
    ground_truth_labels_arr = []
    vocalization_arr = []
    file_indices_arr = []
    dataset_indices_arr = []
    predictions_arr = []

    # Initialize file mapping
    file_map = {}
    current_file_index = 0

    print("Initialized empty arrays")
    print("-" * 50)

    samples = int(samples)
    total_samples = 0
    samples_per_dataset = samples // len(data_dirs)

    for dataset_idx, data_dir in enumerate(data_dirs):
        data_loader = load_data(data_dir=data_dir, context=context)
        data_loader_iter = iter(data_loader)
        dataset_samples = 0

        while dataset_samples < samples_per_dataset:
            try:
                data, ground_truth_label, vocalization, file_paths = next(data_loader_iter)
                print(f"\nProcessing new sample batch")
                print("-" * 50)

                if data.shape[1] < 100:
                    continue

                print("Initial tensor shapes:")
                print(f"data: {data.shape}")
                print(f"ground_truth_label: {ground_truth_label.shape}")
                print(f"vocalization: {vocalization.shape}")
                print("-" * 30)

                # Handle file paths - take first one since we're using batch size 1
                current_file_path = file_paths[0] if isinstance(file_paths, list) else file_paths

                if current_file_path not in file_map:
                    file_map[current_file_index] = current_file_path
                    # Convert tensors to numpy arrays with correct dimensions
                    data_np = data.cpu().numpy()  # [1, 500, 513]
                    ground_truth_np = ground_truth_label.cpu().numpy()[0]  # [500]
                    vocalization_np = vocalization.cpu().numpy()[0]  # [500]
                    
                    print("After numpy conversion:")
                    print(f"data_np: {data_np.shape}")
                    print(f"ground_truth_np: {ground_truth_np.shape}")
                    print(f"vocalization_np: {vocalization_np.shape}")
                    print("-" * 30)
                    
                    # Create file and dataset indices matching ground truth shape
                    file_indices = np.full_like(ground_truth_np, current_file_index)  # [500]
                    dataset_indices = np.full_like(ground_truth_np, dataset_idx)  # [500]

                    print("Created indices arrays:")
                    print(f"file_indices: {file_indices.shape}")
                    print(f"dataset_indices: {dataset_indices.shape}")
                    print("-" * 30)

                    # Store results
                    spec_to_store = data_np[0]  # [500, 513]
                    print("Shapes being stored:")
                    print(f"spec_to_store: {spec_to_store.shape}")
                    print(f"ground_truth_np: {ground_truth_np.shape}")
                    print(f"vocalization_np: {vocalization_np.shape}")
                    print(f"file_indices: {file_indices.shape}")
                    print("-" * 30)

                    spec_arr.append(spec_to_store)
                    ground_truth_labels_arr.append(ground_truth_np)
                    vocalization_arr.append(vocalization_np)
                    file_indices_arr.append(file_indices)
                    dataset_indices_arr.append(dataset_indices)

                    if not raw_spectogram:
                        # shape: (B,T,F), we want to produce embeddings from model
                        # model expects: (B,1,T,F) -> inference_forward -> (B,T,H)
                        data_in = data.permute(0, 2, 1).unsqueeze(1)  # (B,1,T,F)
                        # Run inference
                        _, layers = model.inference_forward(data_in.to(device))
                        
                        # Extract embeddings from the specified layer_index and dict_key
                        if layer_index is None:
                            layer_index = -1
                        if dict_key is None:
                            dict_key = "attention_output"
                        
                        layer_output_dict = layers[layer_index]
                        output = layer_output_dict.get(dict_key, None)
                        if output is None:
                            print(f"Invalid key: {dict_key}. Skipping this batch.")
                            continue

                        # output: (B,T,H)
                        batches, time_bins, features = output.shape
                        predictions = output.reshape(batches * time_bins, features)
                        predictions_arr.append(predictions.detach().cpu().numpy())

                    current_file_index += 1
                else:
                    # if we already processed this file, skip it
                    continue

                print("Current array lengths:")
                print(f"spec_arr: {len(spec_arr)}")
                print(f"ground_truth_labels_arr: {len(ground_truth_labels_arr)}")
                print(f"vocalization_arr: {len(vocalization_arr)}")
                print(f"file_indices_arr: {len(file_indices_arr)}")
                print("-" * 50)

                total_samples += data.shape[0]
                dataset_samples += data.shape[0]

                if len(spec_arr) >= samples:
                    break

                # Check total accumulated time points
                # To 
                total_timepoints = sum(len(v) for v in vocalization_arr) if vocalization_arr else 0
                if total_timepoints >= samples:
                    print(f"Reached {total_timepoints} time points (requested {samples}). Breaking...")
                    break

            except StopIteration:
                print(f"Dataset {data_dir} exhausted after {dataset_samples} samples")
                break

    print("\nFinal array lengths before concatenation:")
    print(f"spec_arr: {len(spec_arr)}")
    print(f"ground_truth_labels_arr: {len(ground_truth_labels_arr)}")
    print(f"vocalization_arr: {len(vocalization_arr)}")
    print(f"file_indices_arr: {len(file_indices_arr)}")
    print("-" * 50)

    print("\nSample shapes from arrays before concatenation:")
    print(f"spec_arr[0]: {spec_arr[0].shape}")
    print(f"ground_truth_labels_arr[0]: {ground_truth_labels_arr[0].shape}")
    print(f"vocalization_arr[0]: {vocalization_arr[0].shape}")
    print(f"file_indices_arr[0]: {file_indices_arr[0].shape}")
    print("-" * 50)

    # Concatenate arrays along first dimension
    spec = np.stack(spec_arr, axis=0)
    ground_truth_labels = np.concatenate(ground_truth_labels_arr, axis=0)
    vocalization = np.concatenate(vocalization_arr, axis=0)
    file_indices = np.concatenate(file_indices_arr, axis=0)
    dataset_indices = np.concatenate(dataset_indices_arr, axis=0)

    print("\nFinal concatenated shapes:")
    print(f"spec: {spec.shape}")
    print(f"ground_truth_labels: {ground_truth_labels.shape}")
    print(f"vocalization: {vocalization.shape}")
    print(f"file_indices: {file_indices.shape}")
    print("-" * 50)

    if not raw_spectogram:
        predictions = np.concatenate(predictions_arr, axis=0)
    else:
        predictions = spec

    # Create directory for visualizations
    experiment_dir = os.path.join("imgs", "umap_plots", save_name if save_name else "umap_experiment")
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    
    # Create subdirectory for individual spectrograms
    spec_dir = os.path.join(experiment_dir, "individual_specs")
    if not os.path.exists(spec_dir):
        os.makedirs(spec_dir)

    # Plot individual spectrograms (first 20 only)
    if len(spec_arr) < 20:
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(data[0].cpu().numpy(), aspect='auto', origin='lower')
        ax.set_title(f"Spectrogram {len(spec_arr)}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(spec_dir, f"spec_{len(spec_arr)}.png"))
        plt.close(fig)

    # After concatenation, plot the concatenated view correctly
    fig1, ax1 = plt.subplots(figsize=(20, 10))
    # Reshape spec to be (Time, Frequency) for visualization
    # Take first 1000 time points if available
    num_timepoints = min(1000, spec.shape[0] * spec.shape[2])
    spec_vis = spec[:num_timepoints//spec.shape[2]].transpose(0, 2, 1).reshape(-1, spec.shape[1])
    im1 = ax1.imshow(spec_vis.T, aspect='auto', origin='lower')
    ax1.set_title("First 1000 Time Points of Spectrograms", fontsize=16)
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    plt.colorbar(im1, ax=ax1)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "raw_spectrograms.png"))
    plt.close(fig1)

    # Plot neural states
    if not raw_spectogram:
        fig2, ax2 = plt.subplots(figsize=(20, 10))
        im2 = ax2.imshow(predictions[:1000].T, aspect='auto', origin='lower')
        ax2.set_title("First 1000 Neural States", fontsize=16)
        ax2.set_xlabel("Time", fontsize=12)
        ax2.set_ylabel("Feature Dimension", fontsize=12)
        plt.colorbar(im2, ax=ax2)
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, "raw_neural_states.png"))
        plt.close(fig2)

    # Skip vocalization filtering
    if samples > len(predictions):
        samples = len(predictions)
    else:
        predictions = predictions[:samples]
        ground_truth_labels = ground_truth_labels[:samples]
        spec = spec[:samples]
        file_indices = file_indices[:samples]
        dataset_indices = dataset_indices[:samples]
        vocalization = vocalization[:samples]

    print("Initializing UMAP reducer...")
    reducer = umap.UMAP(n_neighbors=200, min_dist=0, n_components=2, metric='euclidean')
    print("UMAP reducer initialized.")
    embedding_outputs = reducer.fit_transform(predictions)
    print("UMAP fitting complete. Shape of embedding outputs:", embedding_outputs.shape)

    print("Generating HDBSCAN labels...")
    hdbscan_labels = generate_hdbscan_labels(embedding_outputs, min_samples=1, min_cluster_size=500)
    print("HDBSCAN labels generated. Unique labels found:", np.unique(hdbscan_labels))

    unique_clusters = np.unique(hdbscan_labels)
    unique_ground_truth_labels = np.unique(ground_truth_labels)

    def create_color_palette(n_colors):
        colors = glasbey.create_palette(palette_size=n_colors)
        def hex_to_rgb(hex_str):
            hex_str = hex_str.lstrip('#')
            return tuple(int(hex_str[i:i+2], 16)/255.0 for i in (0,2,4))
        rgb_colors = [hex_to_rgb(color) for color in colors]
        return rgb_colors

    n_ground_truth_labels = len(unique_ground_truth_labels)
    n_hdbscan_clusters = len(unique_clusters)
    ground_truth_colors = create_color_palette(n_ground_truth_labels)
    hdbscan_colors = create_color_palette(n_hdbscan_clusters)
    
    # Ensure label 0 is black for both color palettes
    ground_truth_colors[0] = (0, 0, 0)
    hdbscan_colors[0] = (0, 0, 0)

    experiment_dir = os.path.join("imgs", "umap_plots", save_name if save_name else "umap_experiment")
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    fig1, ax1 = plt.subplots(figsize=(16,16), facecolor='white')
    ax1.set_facecolor('white')
    ax1.scatter(embedding_outputs[:,0], embedding_outputs[:,1],
                c=hdbscan_labels, s=10, alpha=0.1,
                cmap=mcolors.ListedColormap(hdbscan_colors))
    ax1.set_title("HDBSCAN Discovered Labels", fontsize=48)
    ax1.set_xlabel('UMAP Dimension 1', fontsize=48)
    ax1.set_ylabel('UMAP Dimension 2', fontsize=48)
    ax1.tick_params(axis='both', which='both', bottom=False, left=False,
                    labelbottom=False, labelleft=False)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "hdbscan_labels.png"),
                facecolor=fig1.get_facecolor(), edgecolor='none')
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(16,16), facecolor='white')
    ax2.set_facecolor('white')
    ax2.scatter(embedding_outputs[:,0], embedding_outputs[:,1],
                c=ground_truth_labels, s=10, alpha=0.1,
                cmap=mcolors.ListedColormap(ground_truth_colors))
    ax2.set_title("Ground Truth Labels", fontsize=48)
    ax2.set_xlabel('UMAP Dimension 1', fontsize=48)
    ax2.set_ylabel('UMAP Dimension 2', fontsize=48)
    ax2.tick_params(axis='both', which='both', bottom=False, left=False,
                    labelbottom=False, labelleft=False)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "ground_truth_labels.png"),
                facecolor=fig2.get_facecolor(), edgecolor='none')
    plt.close(fig2)

    np.savez(
        f"files/{save_name if save_name else 'default_experiment'}",
        embedding_outputs=embedding_outputs,
        hdbscan_labels=hdbscan_labels,
        ground_truth_labels=ground_truth_labels,
        predictions=predictions,
        s=spec,
        hdbscan_colors=hdbscan_colors,
        ground_truth_colors=ground_truth_colors,
        original_spectogram=spec,
        vocalization=vocalization,
        file_indices=file_indices,
        dataset_indices=dataset_indices,
        file_map=file_map
    )

def main(experiment_folder, data_dirs, category_colors_file, save_name, samples, layer_index, dict_key, context, raw_spectogram, save_dict_for_analysis, min_cluster_size):
    device = torch.device('cpu')
    model = load_model(experiment_folder)
    model = model.to(device)

    plot_umap_projection(
        model=model, 
        device=device, 
        data_dirs=data_dirs,
        samples=samples, 
        category_colors_file=category_colors_file, 
        layer_index=layer_index, 
        dict_key=dict_key, 
        context=context, 
        raw_spectogram=raw_spectogram,
        save_dict_for_analysis=save_dict_for_analysis,
        save_name=save_name,
        min_cluster_size=min_cluster_size
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dimension-reduced birdsong plots using BirdJEPA.")
    parser.add_argument('--experiment_folder', type=str, default="experiments/default_experiment", help='Path to the experiment folder.')
    parser.add_argument('--data_dirs', nargs='+', default=["train_dir"], help='List of directories containing the data.')
    parser.add_argument('--category_colors_file', type=str, default="files/category_colors_llb3.pkl", help='Path to the category colors file.')
    parser.add_argument('--save_name', type=str, default="default_experiment_hdbscan", help='Name to save the output.')
    parser.add_argument('--samples', type=float, default=1e6, help='Number of samples to use.')
    parser.add_argument('--layer_index', type=int, default=-2, help='Layer index to use for UMAP projection.')
    parser.add_argument('--dict_key', type=str, default="attention_output", help='Dictionary key to use for UMAP projection.')
    parser.add_argument('--context', type=int, default=1000, help='Context size for the model.')
    parser.add_argument('--raw_spectogram', action='store_true', help='Whether to use raw spectogram.')
    parser.add_argument('--save_dict_for_analysis', action='store_true', help='Whether to save dictionary for analysis.')
    parser.add_argument('--min_cluster_size', type=int, default=500, help='Minimum cluster size for HDBSCAN.')

    args = parser.parse_args()

    main(
        args.experiment_folder, 
        args.data_dirs,
        args.category_colors_file, 
        args.save_name,
        args.samples,
        args.layer_index,
        args.dict_key,
        args.context,
        args.raw_spectogram,
        args.save_dict_for_analysis,
        args.min_cluster_size
    )
