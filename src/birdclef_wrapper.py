#!/usr/bin/env python3
"""
Simple wrapper for the WavtoSpec class that adds BirdCLEF-specific functionality.
"""
import os
import argparse
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from spectrogram_generator import WavtoSpec

def process_files(src_dir, dst_dir, step_size, nfft, train_csv, max_files=None, random_subset=False, single_threaded=False):
    """Process audio files for BirdCLEF dataset"""
    
    # Load the CSV to get primary_label mapping
    df = pd.read_csv(train_csv)
    print(f"Loaded metadata for {len(df)} audio files from {train_csv}")
    
    # Create a mapping from directory name to primary_label
    dir_to_primary_label = {}
    
    # Extract the directory name from filename and map to primary_label
    for _, row in df.iterrows():
        # CSV filename format: "1139490/CSA36385.ogg"
        parts = row['filename'].split('/')
        if len(parts) >= 2:
            # Extract file ID without extension
            file_id = os.path.splitext(parts[1])[0]
            # Map it to primary_label
            dir_to_primary_label[file_id] = row['primary_label']
    
    # Print some example mappings
    print("Example mappings from file ID to primary_label:")
    count = 0
    for file_id, primary_label in dir_to_primary_label.items():
        print(f"  {file_id} -> {primary_label}")
        count += 1
        if count >= 5:
            break
    
    # Find all audio files
    all_files = []
    for ext in ['*.ogg', '*.wav', '*.mp3']:
        all_files.extend(glob.glob(os.path.join(src_dir, '**', ext), recursive=True))
    
    print(f"Found {len(all_files)} audio files in {src_dir}")
    
    # Limit files if needed
    if max_files is not None and max_files > 0:
        if random_subset:
            all_files = random.sample(all_files, min(max_files, len(all_files)))
        else:
            all_files = all_files[:max_files]
        print(f"Limited to {len(all_files)} files")
    
    # Create spectrogram generator
    generator = WavtoSpec(
        src_dir=src_dir,
        dst_dir=dst_dir,
        step_size=step_size,
        nfft=nfft,
        single_threaded=single_threaded
    )
    
    # Process files
    skipped = 0
    processed = 0
    
    for file_path in tqdm(all_files, desc="Processing audio files"):
        try:
            # Check if file is too short
            import librosa
            try:
                duration = librosa.get_duration(path=file_path)
            except:
                # Fallback method
                import soundfile as sf
                info = sf.info(file_path)
                duration = info.duration
                
            if duration < 3.0:
                print(f"Skipping {file_path} - too short ({duration:.2f}s < 3.0s)")
                skipped += 1
                continue
            
            # Extract filename and get the correct primary_label
            file_name = os.path.basename(file_path)
            file_id = os.path.splitext(file_name)[0]
            
            # Try to find the primary_label for this file
            # First from the CSV mapping
            if file_id in dir_to_primary_label:
                species_id = dir_to_primary_label[file_id]
            else:
                # Use the directory name as fallback - this is what we did before
                species_id = os.path.basename(os.path.dirname(file_path))
                
                # Print a message for the first few unmapped files
                if processed + skipped < 10:
                    print(f"File {file_id} not found in CSV, using directory name: {species_id}")
            
            # Generate spectrogram
            spec, results, _ = generator.convert_to_spectrogram(
                file_path, 
                song_detection_json_path=None, 
                save_npz=False
            )
            
            if spec is None or results is None:
                skipped += 1
                continue
                
            # Save with custom filename that includes species ID
            for i, (start_sec, end_sec) in enumerate(results):
                segment_start = int(np.searchsorted(
                    np.arange(spec.shape[1]) * step_size / 32000, start_sec))
                segment_end = int(np.searchsorted(
                    np.arange(spec.shape[1]) * step_size / 32000, end_sec))
                    
                # Extract segment
                segment = spec[:, segment_start:segment_end].copy()
                
                # Create output filename with species ID
                output_path = os.path.join(
                    dst_dir, 
                    f"{species_id}_{file_id}_segment_{i}.npz"
                )
                
                # Create dummy labels
                segment_labels = np.zeros(segment_end - segment_start)
                
                # Convert to 8-bit representation
                # First normalize to 0-1 range
                min_val = segment.min()
                max_val = segment.max()
                
                # Avoid division by zero
                if max_val > min_val:
                    segment_uint8 = np.uint8(255 * (segment - min_val) / (max_val - min_val))
                else:
                    segment_uint8 = np.zeros_like(segment, dtype=np.uint8)
                
                # Save with species ID included in the data - uncompressed
                np.savez(  # Use uncompressed version
                    output_path,
                    s=segment_uint8,
                    labels=segment_labels,
                    sr=32000,
                    hop_length=step_size,
                    file_name=file_name,
                    species_id=species_id,
                    min_val=min_val,  # Store scaling info for reconstruction
                    max_val=max_val
                )
            
            processed += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            skipped += 1
    
    print(f"Successfully processed {processed}/{len(all_files)} files")
    print(f"Spectrograms saved to {dst_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BirdCLEF audio files into spectrograms")
    parser.add_argument("--src_dir", type=str, required=True, help="Directory containing audio files organized by species")
    parser.add_argument("--dst_dir", type=str, required=True, help="Directory to save spectrograms")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train.csv for metadata")
    parser.add_argument("--step_size", type=int, default=119, help="Step size for spectrogram generation")
    parser.add_argument("--nfft", type=int, default=1024, help="Number of FFT points")
    parser.add_argument("--single_threaded", action="store_true", help="Use single threaded processing")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to process (for testing)")
    parser.add_argument("--random_subset", action="store_true", help="Select a random subset of files instead of first N")
    
    args = parser.parse_args()
    
    process_files(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        step_size=args.step_size,
        nfft=args.nfft,
        train_csv=args.train_csv,
        max_files=args.max_files,
        random_subset=args.random_subset,
        single_threaded=args.single_threaded
    ) 