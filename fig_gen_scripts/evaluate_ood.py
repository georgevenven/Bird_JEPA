#!/usr/bin/env python3
"""
evaluate_ood.py - Evaluate trained Bird JEPA model on out-of-distribution data

This script loads a trained model, applies masking, makes predictions, and calculates
average MSE to evaluate performance on OOD datasets.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path to fix imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import asdict
import matplotlib.pyplot as plt

# Import model components
from models.birdjepa import BJConfig, BirdJEPA, SpectrogramDecoder
from data.bird_datasets import TorchSpecDataset

def token_mask_to_pixel_mask(mask_tok, Fp, Tp, f_stride, t_stride):
    """
    Converts a token mask (B, Fp*Tp) or (Fp*Tp) to a pixel mask.
    """
    if mask_tok.ndim == 1:
        B = 1
        mask_tok = mask_tok.unsqueeze(0)
    else:
        B = mask_tok.shape[0]

    F_bins = Fp * f_stride
    T_bins = Tp * t_stride
    device = mask_tok.device

    # Reshape token mask to grid
    mask_tok_grid = mask_tok.view(B, Fp, Tp)

    # Create patch template
    patch_template = torch.ones((f_stride, t_stride), device=device, dtype=torch.bool)

    # Use broadcasting and repeat for efficiency
    mask_tok_expanded = mask_tok_grid.unsqueeze(2).unsqueeze(4)
    patch_template_expanded = patch_template.unsqueeze(0).unsqueeze(1).unsqueeze(3)
    pixel_mask_expanded = mask_tok_expanded & patch_template_expanded
    pixel_mask = pixel_mask_expanded.view(B, Fp, Tp, f_stride, t_stride).permute(0, 1, 3, 2, 4).reshape(B, F_bins, T_bins)

    return pixel_mask

def generate_batch_mask(F_tokens, T_tokens, mask_ratio=0.75, device="cpu"):
    """Generate random mask for a batch of tokens - same as pretrain.py"""
    num_tokens = F_tokens * T_tokens
    num_masked = int(mask_ratio * num_tokens)

    # Generate random indices to mask ONCE
    indices = torch.randperm(num_tokens, device=device)
    masked_indices = indices[:num_masked]

    # Create boolean mask (True where masked)
    mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    mask[masked_indices] = True
    
    return mask

def load_model_and_config(checkpoint_path, config_path, model_config_path, device):
    """Load trained model from checkpoint."""
    # Load configurations
    with open(config_path, 'r') as f:
        train_config = json.load(f)
    
    with open(model_config_path, 'r') as f:
        model_config_dict = json.load(f)
    
    # Create model config
    cfg = BJConfig(**model_config_dict)
    
    # Initialize models
    encoder = BirdJEPA(cfg)
    
    # Calculate patch dimensions
    Fp = encoder.Fp
    Tp = encoder.Tp
    f_stride = cfg.n_mels // Fp
    t_stride = train_config['context_length'] // Tp
    
    decoder = SpectrogramDecoder(cfg, 
                                patch_size_f=f_stride,
                                patch_size_t=t_stride,
                                in_chans=1)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    encoder.load_state_dict(checkpoint['enc'])
    decoder.load_state_dict(checkpoint['dec'])
    
    # Move to device
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder, cfg, train_config, Fp, Tp, f_stride, t_stride

@torch.no_grad()
def evaluate_model(encoder, decoder, dataset, cfg, train_config, Fp, Tp, f_stride, t_stride, 
                  mask_ratio=0.75, num_samples=None, device='cuda', save_visualizations=False, output_dir=None):
    """
    Evaluate model on dataset and return average MSE.
    Uses the same masking procedure as training (single mask per batch).
    """
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=16,  # Smaller batch size for evaluation
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    total_mse = 0.0
    total_samples = 0
    mse_losses = []
    
    # Setup visualization directory
    if save_visualizations and output_dir:
        vis_dir = Path(output_dir) / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to: {vis_dir}")
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    for batch_idx, (spec_batch, _, filenames) in enumerate(dataloader):
        if num_samples is not None and total_samples >= num_samples:
            break
            
        spec_batch = spec_batch.to(device)
        B, freq_bins, time_bins = spec_batch.shape
        
        # Add channel dimension: (B, freq_bins, time_bins) -> (B, 1, freq_bins, time_bins)
        spec_batch = spec_batch.unsqueeze(1)
        
        # Generate ONE mask for the entire batch (same as training)
        # This matches the training procedure from pretrain.py
        num_tokens = Fp * Tp
        single_mask_tok_flat = generate_batch_mask(Fp, Tp, mask_ratio, device)
        # Repeat the same mask for all items in the batch
        mask_batch = single_mask_tok_flat.unsqueeze(0).expand(B, -1)  # (B, Fp*Tp)
        
        # Calculate indices based on the single mask (same as training)
        num_masked = single_mask_tok_flat.sum().item()
        num_visible = num_tokens - num_masked
        ids_shuffle = torch.argsort(torch.rand(num_tokens, device=device))
        ids_keep = ids_shuffle[:num_visible]  # Indices of visible tokens
        
        # Encode full spectrogram (same as training)
        encoded_all, _, _ = encoder(spec_batch)  # (B, Fp*Tp, D)
        D_enc = encoded_all.shape[-1]
        
        # Select only VISIBLE encoded tokens using indices (same as training)
        ids_keep_batch = ids_keep.unsqueeze(0).expand(B, -1)  # (B, num_visible)
        encoded_vis = torch.gather(encoded_all, dim=1, index=ids_keep_batch.unsqueeze(-1).expand(-1, -1, D_enc))
        # encoded_vis shape: (B, num_visible, D_enc)
        
        # Decode (reconstruct masked patches) - same as training
        pred_pixels = decoder(encoded_vis, mask_batch, (Fp, Tp))  # (B * num_masked, patch_pixels)
        
        # Get target pixels for masked patches - same as training
        pixel_mask = token_mask_to_pixel_mask(mask_batch, Fp, Tp, f_stride, t_stride)  # (B, freq_bins, time_bins)
        
        # Extract patches from the original spectrogram (same as training)
        spec_patches = spec_batch.unfold(2, f_stride, f_stride).unfold(3, t_stride, t_stride)
        # spec_patches shape: (B, 1, Fp, Tp, f_stride, t_stride)
        spec_patches = spec_patches.permute(0, 2, 3, 1, 4, 5).reshape(B, Fp * Tp, -1)
        # spec_patches shape: (B, Fp*Tp, num_pixels_per_patch)
        
        # Select the target patches using the boolean mask (same as training)
        target_pixels_flat = spec_patches[mask_batch]
        # target_pixels_flat shape: (B * num_masked, num_pixels_per_patch)
        
        # Create reconstructed spectrograms for visualization
        if save_visualizations and output_dir:
            reconstructed_specs = create_reconstructed_spectrograms(
                spec_batch, pred_pixels, pixel_mask, mask_batch, Fp, Tp, f_stride, t_stride
            )
        
        # Calculate MSE for the entire batch (same as training)
        if pred_pixels.shape[0] > 0 and target_pixels_flat.shape[0] > 0:
            mse = F.mse_loss(pred_pixels, target_pixels_flat, reduction='mean')
            mse_losses.append(mse.item())
            total_mse += mse.item()
            total_samples += B  # Count number of samples, not batches
        
        # Save visualizations for first few batches
        if save_visualizations and output_dir and batch_idx < 5:  # Save first 5 batches
            save_batch_visualizations(
                spec_batch, reconstructed_specs, pixel_mask, 
                filenames, vis_dir, batch_idx, mse_losses[-1] if mse_losses else 0.0
            )
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1} batches, {total_samples} samples")
    
    # Calculate statistics
    avg_mse = total_mse / len(mse_losses) if len(mse_losses) > 0 else 0.0
    std_mse = np.std(mse_losses) if len(mse_losses) > 0 else 0.0
    
    results = {
        'avg_mse': avg_mse,
        'std_mse': std_mse,
        'total_samples': total_samples,
        'mask_ratio': mask_ratio,
        'all_mse_values': mse_losses
    }
    
    return results

def create_reconstructed_spectrograms(spec_batch, pred_pixels, pixel_mask, mask_batch, Fp, Tp, f_stride, t_stride):
    """
    Create reconstructed spectrograms by filling in predicted pixels.
    Updated for batch-consistent masking where all samples share the same mask pattern.
    """
    B = spec_batch.shape[0]
    reconstructed = spec_batch.clone()
    
    # Since all samples in the batch have the same mask pattern, we can use the first mask
    single_mask = mask_batch[0].view(Fp, Tp)  # (Fp, Tp) - same pattern for all samples
    
    # Calculate how many patches are masked (same for all samples)
    num_masked_patches = single_mask.sum().item()
    patches_per_sample = num_masked_patches
    
    # pred_pixels is organized as: [sample0_patch0, sample0_patch1, ..., sample1_patch0, sample1_patch1, ...]
    # Shape: (B * num_masked_patches, patch_pixels)
    
    for b in range(B):
        # Calculate the starting index for this sample's predictions
        sample_start_idx = b * patches_per_sample
        patch_idx = 0
        
        # Iterate through the token grid to find masked patches
        for fp in range(Fp):
            for tp in range(Tp):
                if single_mask[fp, tp]:  # This patch is masked (same pattern for all samples)
                    # Get patch boundaries
                    f_start = fp * f_stride
                    f_end = f_start + f_stride
                    t_start = tp * t_stride
                    t_end = t_start + t_stride
                    
                    # Get predicted pixels for this patch from this sample
                    pred_idx = sample_start_idx + patch_idx
                    patch_pixels = pred_pixels[pred_idx].view(f_stride, t_stride)
                    
                    # Fill in the reconstruction
                    reconstructed[b, 0, f_start:f_end, t_start:t_end] = patch_pixels
                    
                    patch_idx += 1
    
    return reconstructed

def save_batch_visualizations(spec_batch, reconstructed_specs, pixel_mask, filenames, vis_dir, batch_idx, batch_mse):
    """
    Save visualization images for a batch.
    """
    B = spec_batch.shape[0]
    
    # Save up to 4 samples per batch
    num_to_save = min(4, B)
    
    for i in range(num_to_save):
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Set larger font sizes
        plt.rcParams.update({'font.size': 24})
        
        # Original spectrogram
        original = spec_batch[i, 0].cpu().numpy()
        im1 = axes[0].imshow(original, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title(f'A. Original\n{filenames[i]}', fontsize=24)
        axes[0].set_ylabel('Frequency bins', fontsize=24)
        axes[0].tick_params(labelsize=20)
        
        # Masked spectrogram (set masked regions to 0)
        masked = original.copy()
        mask = pixel_mask[i].cpu().numpy()
        masked[mask] = 0
        im2 = axes[1].imshow(masked, aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title(f'B. Masked (75% hidden)', fontsize=24)
        axes[1].set_ylabel('Frequency bins', fontsize=24)
        axes[1].tick_params(labelsize=20)
        
        # Reconstructed spectrogram
        reconstructed = reconstructed_specs[i, 0].cpu().numpy()
        im3 = axes[2].imshow(reconstructed, aspect='auto', origin='lower', cmap='viridis')
        axes[2].set_title(f'C. Reconstructed\nMSE: {batch_mse:.4f}', fontsize=24)
        axes[2].set_ylabel('Frequency bins', fontsize=24)
        axes[2].tick_params(labelsize=20)
        
        # Set x-label only on bottom plot
        axes[2].set_xlabel('Time bins', fontsize=24)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'batch_{batch_idx:03d}_sample_{i:02d}.png'
        plt.savefig(vis_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Reset font size
        plt.rcParams.update({'font.size': 10})
    
    print(f"Saved {num_to_save} visualizations for batch {batch_idx}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Bird JEPA model on OOD data')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--run_dir', type=str, required=True,
                       help='Path to run directory containing config files')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to evaluation dataset directory')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                       help='Fraction of tokens to mask (default: 0.75)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (default: all)')
    parser.add_argument('--context_length', type=int, default=None,
                       help='Context length override (default: use from config)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file to save results (default: print only)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save reconstruction visualization images')
    parser.add_argument('--vis_output_dir', type=str, default='./evaluation_output',
                       help='Directory to save visualizations (default: ./evaluation_output)')
    
    args = parser.parse_args()
    
    # Setup paths
    run_dir = Path(args.run_dir)
    config_path = run_dir / 'config.json'
    model_config_path = run_dir / 'model_config.json'
    
    # Check if files exist
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    if not Path(args.data_dir).exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    encoder, decoder, cfg, train_config, Fp, Tp, f_stride, t_stride = load_model_and_config(
        args.checkpoint, config_path, model_config_path, device
    )
    
    print(f"Model loaded successfully!")
    print(f"Token grid: {Fp} x {Tp}")
    print(f"Patch size: {f_stride} x {t_stride}")
    
    # Load dataset
    context_length = args.context_length or train_config['context_length']
    print(f"Loading dataset from {args.data_dir} with context_length={context_length}")
    
    dataset = TorchSpecDataset(
        args.data_dir,
        segment_len=context_length,
        infinite=False  # Don't use infinite sampling for evaluation
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Evaluate model
    print(f"Starting evaluation with mask_ratio={args.mask_ratio}")
    if args.save_visualizations:
        print(f"Visualizations will be saved to: {args.vis_output_dir}/visualizations/")
    
    results = evaluate_model(
        encoder, decoder, dataset, cfg, train_config, 
        Fp, Tp, f_stride, t_stride,
        mask_ratio=args.mask_ratio,
        num_samples=args.num_samples,
        device=device,
        save_visualizations=args.save_visualizations,
        output_dir=args.vis_output_dir
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Dataset: {args.data_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples evaluated: {results['total_samples']}")
    print(f"Mask ratio: {results['mask_ratio']:.2f}")
    print(f"Average MSE: {results['avg_mse']:.6f}")
    print(f"MSE Std Dev: {results['std_mse']:.6f}")
    if args.save_visualizations:
        print(f"Visualizations saved to: {args.vis_output_dir}/visualizations/")
    print("="*50)
    
    # Save results if requested
    if args.output_file:
        output_data = {
            'dataset_path': args.data_dir,
            'checkpoint_path': args.checkpoint,
            'run_dir': args.run_dir,
            'evaluation_results': results,
            'config': {
                'mask_ratio': args.mask_ratio,
                'num_samples': args.num_samples,
                'context_length': context_length,
                'device': args.device,
                'save_visualizations': args.save_visualizations,
                'vis_output_dir': args.vis_output_dir
            }
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main() 