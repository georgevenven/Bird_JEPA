#!/usr/bin/env python
import os
import shutil
import argparse
import zipfile
import tempfile
import re
from pathlib import Path
import datetime

def create_zip_archive(experiment_path, output_dir="zips"):
    """
    Create a zip archive of the project including src, scripts, shell_scripts directories
    and the specified experiment with only the best weights.
    
    Args:
        experiment_path (str): Path to the experiment directory
        output_dir (str): Directory to save the zip file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure experiment path exists
    experiment_path = Path(experiment_path)
    if not experiment_path.exists():
        raise ValueError(f"Experiment path not found: {experiment_path}")
    
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
            if os.path.exists(directory):
                shutil.copytree(directory, temp_path / directory)
                print(f"Added directory: {directory}")
        
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
        
        # Create zip archive
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
    
    print(f"Archive created successfully: {zip_filename}")
    return zip_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a project export zip with essential files")
    parser.add_argument("experiment_path", help="Path to the experiment directory")
    parser.add_argument("--output-dir", default="zips", help="Directory to save the zip file (default: 'zips')")
    
    args = parser.parse_args()
    
    try:
        zip_path = create_zip_archive(args.experiment_path, args.output_dir)
        print(f"\nExport complete! Archive saved to: {zip_path}")
    except Exception as e:
        print(f"Error creating export: {e}") 