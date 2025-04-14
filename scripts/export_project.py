#!/usr/bin/env python
import os
import shutil
import argparse
import zipfile
import tempfile
import re
from pathlib import Path
import datetime
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Parameters to copy and paste (modify these variables instead of using command line)
experiment_path = "/home/george-vengrovski/Documents/projects/Bird_JEPA/experiments/BirdJEPA_Small_Finetune"  # Path to the experiment directory
output_dir = "/home/george-vengrovski/Documents/projects/Bird_JEPA/zips"  # Directory to save the zip file (default: 'zips')
pretrained_path = "/home/george-vengrovski/Documents/projects/Bird_JEPA/experiments/BirdJEPA_Small_Untrained"  # Optional: Path to pretrained model directory or file

def create_zip_archive(experiment_path, output_dir="zips", pretrained_path=None):
    """
    Create a zip archive of the project including src, scripts, shell_scripts directories
    and the specified experiment with only the best weights.
    
    Args:
        experiment_path (str): Path to the experiment directory
        output_dir (str): Directory to save the zip file
        pretrained_path (str, optional): Path to the pretrained model directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure experiment path exists
    experiment_path = Path(experiment_path)
    if not experiment_path.exists():
        raise ValueError(f"Experiment path not found: {experiment_path}")
    
    # Check pretrained path if provided
    if pretrained_path:
        pretrained_path = Path(pretrained_path)
        if not pretrained_path.exists():
            raise ValueError(f"Pretrained path not found: {pretrained_path}")
    
    # Determine the project root directory
    if 'experiments' in str(experiment_path):
        # Extract project root from experiment path
        project_root = Path(str(experiment_path).split('experiments')[0])
    else:
        # Default to current directory if unable to determine project root
        project_root = Path(os.getcwd())
    
    print(f"Using project root: {project_root}")
    
    # Generate timestamp for zip filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = experiment_path.name
    zip_filename = f"{output_dir}/{experiment_name}_{timestamp}.zip"
    
    print(f"Creating archive: {zip_filename}")
    
    # Create temporary directory for staging files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy directories to temp directory
        for directory in ["src", "scripts", "shell_scripts"]:
            src_dir = project_root / directory
            if src_dir.exists():
                dst_dir = temp_path / directory
                shutil.copytree(src_dir, dst_dir)
                print(f"Added directory: {directory}")
            else:
                print(f"Warning: Directory not found: {src_dir}")
        
        # Create experiment directory in temp
        exp_temp_path = temp_path / "experiment"
        os.makedirs(exp_temp_path, exist_ok=True)
        
        # Copy experiment config files
        for file in experiment_path.glob("*.json"):
            shutil.copy2(file, exp_temp_path)
            print(f"Added config file: {file.name}")
            
        # Copy experiment plots
        for file in experiment_path.glob("*.png"):
            shutil.copy2(file, exp_temp_path)
            print(f"Added plot: {file.name}")
        
        # Find and copy only the best weights
        weights_dir = experiment_path / "weights"
        if weights_dir.exists():
            exp_weights_dir = exp_temp_path / "weights"
            os.makedirs(exp_weights_dir, exist_ok=True)
            
            # Find best weights (files with "best_" prefix)
            best_weight_files = list(weights_dir.glob("best_*_step_*.pt"))
            
            if best_weight_files:
                # Extract step numbers from filenames and sort by numeric value
                def get_step_number(file_path):
                    match = re.search(r'step_(\d+)\.pt', str(file_path))
                    if match:
                        return int(match.group(1))
                    return 0
                
                # Get the best weight file with the highest step number
                best_weight_file = sorted(best_weight_files, key=get_step_number)[-1]
                
                # Copy the best weight file
                shutil.copy2(best_weight_file, exp_weights_dir)
                print(f"Added best weight file: {best_weight_file.name}")
            else:
                print("Warning: No best weight files found!")
        
        # Copy pretrained model if path is provided
        if pretrained_path:
            pretrained_temp_path = temp_path / "pretrained"
            
            if pretrained_path.is_file():
                # If pretrained_path is a file, copy it directly
                os.makedirs(pretrained_temp_path, exist_ok=True)
                shutil.copy2(pretrained_path, pretrained_temp_path)
                print(f"Added pretrained model file: {pretrained_path.name}")
            elif pretrained_path.is_dir():
                # Copy entire directory structure recursively
                shutil.copytree(pretrained_path, pretrained_temp_path)
                print(f"Added all pretrained model files from: {pretrained_path}")
            
        # Create zip archive
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
    
    print(f"Archive created successfully: {zip_filename}")
    return zip_filename

def run_analysis(self):
    print("Running analysis with ONNX Runtime optimization")
    rows = []
    num_samples = len(self.val_wrapper.dataset)
    
    # Convert model to ONNX format (do this once)
    dummy_input = torch.randn(1, 1, 128, 1000)  # Adjust size to match your model input
    torch.onnx.export(
        self.model,
        dummy_input,
        "model.onnx",
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )
    
    # Create ONNX Runtime session
    session_options = ort.SessionOptions()
    session_options.enable_cpu_mem_arena = True
    session_options.intra_op_num_threads = 2  # Match your CPU cores
    session = ort.InferenceSession("model.onnx", session_options)
    
    # Process in batches (use args.batch_size passed from command line)
    batch_size = self.args.batch_size
    all_segments = []
    segment_info = []  # Store metadata for each segment
    
    # Collect segments
    for i in tqdm(range(num_samples), desc="Collecting segments"):
        spec_tensor, label, label_idx, filenames = self.val_wrapper.next_batch()
        if label_idx == -1 or not filenames:
            continue
            
        for j in range(0, spec_tensor.shape[-1], 1000):
            segment = spec_tensor[..., j:j+1000]
            if segment.shape[-1] < 1000:
                pad_size = 1000 - segment.shape[-1]
                pad_tensor = torch.zeros(*segment.shape[:-1], pad_size)
                segment = torch.cat([segment, pad_tensor], dim=-1)
            
            base_id = filenames[0].split("_segment_")[0]
            time_marker = (j // 1000 + 1) * 5
            
            all_segments.append(segment)
            segment_info.append((base_id, time_marker, label_idx))
            
            # Process batch when it reaches batch_size
            if len(all_segments) >= batch_size:
                self._process_batch_onnx(session, all_segments, segment_info, rows, bird_classes)
                all_segments = []
                segment_info = []
    
    # Process remaining segments
    if all_segments:
        self._process_batch_onnx(session, all_segments, segment_info, rows, bird_classes)
    
    # Create submission
    submission_df = pd.DataFrame(rows)
    submission_df.to_csv(self.args.submission_csv, index=False)
    print(f"CSV file '{self.args.submission_csv}' populated with predictions")

def _process_batch_onnx(self, session, segments, segment_info, rows, bird_classes):
    # Stack segments into a batch
    batch = torch.stack(segments).numpy()
    
    # ONNX Runtime inference
    ort_inputs = {session.get_inputs()[0].name: batch}
    ort_outputs = session.run(None, ort_inputs)
    probs = 1/(1 + np.exp(-ort_outputs[0]))  # sigmoid
    
    # Process results
    for i, (base_id, time_marker, _) in enumerate(segment_info):
        row_id = f"{base_id}_{time_marker}"
        row_data = {"row_id": row_id}
        for cls_idx, class_name in enumerate(bird_classes):
            row_data[class_name] = probs[i][cls_idx]
        rows.append(row_data)

if __name__ == "__main__":
    try:
        zip_path = create_zip_archive(experiment_path, output_dir, pretrained_path)
        print(f"\nExport complete! Archive saved to: {zip_path}")
    except Exception as e:
        print(f"Error creating export: {e}")

parser = argparse.ArgumentParser(description="Export project")
parser.add_argument("--use_onnx", action="store_true", help="Use ONNX Runtime for faster inference")

if __name__ == "__main__":
    try:
        zip_path = create_zip_archive(experiment_path, output_dir, pretrained_path)
        print(f"\nExport complete! Archive saved to: {zip_path}")
    except Exception as e:
        print(f"Error creating export: {e}") 