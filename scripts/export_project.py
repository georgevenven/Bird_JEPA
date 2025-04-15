#!/usr/bin/env python
import os
import shutil
import argparse
import zipfile
import tempfile
import re
import sys
import platform
import subprocess
from pathlib import Path
import datetime
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# parameters (modify these variables instead of using command line)
experiment_path = "/home/george-vengrovski/Documents/projects/Bird_JEPA/experiments/BirdJEPA_Small_Finetune"  # experiment directory
output_dir = "/home/george-vengrovski/Documents/projects/Bird_JEPA/zips"  # where to save zip file
pretrained_path = "/home/george-vengrovski/Documents/projects/Bird_JEPA/experiments/BirdJEPA_Small_Untrained"  # pretrained model (dir or file)
download_onnx = False  # download onnxruntime wheel during zipping

# parameters to target a wheel compatible with offline kaggle (adjust to match kaggle's python/abi)
target_onnxruntime_version = "1.13.1"
target_platform_tag = "manylinux2014_x86_64"  # common tag in kaggle env.
target_python_version = "3.8"              # target python version in kaggle env.
target_abi_tag = "cp38"                      # corresponding ABI tag


def download_onnxruntime(temp_dir):
    """
    download the onnxruntime wheel using pip download,
    forcing a target platform (kaggle offline env) by setting:
      --platform target_platform_tag
      --python-version target_python_version
      --abi target_abi_tag
    this is intended to run in an environment with internet access.
    """
    download_dir = os.path.join(temp_dir, "temp_download")
    os.makedirs(download_dir, exist_ok=True)
    
    print("downloading onnxruntime wheel using pip...")
    cmd = [
        sys.executable, "-m", "pip", "download",
        f"onnxruntime=={target_onnxruntime_version}",  # pinned version
        "--no-deps",
        "-d", download_dir,
        "--platform", target_platform_tag,
        "--python-version", target_python_version,
        "--abi", target_abi_tag,
        "--only-binary", ":all:"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"failed to download onnxruntime:\nstdout: {e.stdout}\nstderr: {e.stderr}")
    
    wheel_files = [f for f in os.listdir(download_dir) if f.endswith('.whl') and 'onnxruntime' in f]
    if not wheel_files:
        raise RuntimeError("no onnxruntime wheel file found after pip download.")
    
    wheel_file = os.path.join(download_dir, wheel_files[0])
    dest_file = os.path.join(temp_dir, wheel_files[0])
    shutil.copy2(wheel_file, dest_file)
    print(f"downloaded onnxruntime wheel: {os.path.basename(dest_file)}")
    return dest_file

def create_zip_archive(experiment_path, output_dir="zips", pretrained_path=None, onnx_wheel_path=None, download_onnx=False):
    """
    create a zip archive of the project including source directories and experiment files,
    and include the onnxruntime wheel downloaded during the zipping process.
    the resulting archive is designed for offline kaggle use.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    experiment_path = Path(experiment_path)
    if not experiment_path.exists():
        raise ValueError(f"experiment path not found: {experiment_path}")
    
    if pretrained_path:
        pretrained_path = Path(pretrained_path)
        if not pretrained_path.exists():
            raise ValueError(f"pretrained path not found: {pretrained_path}")
    
    if onnx_wheel_path and not download_onnx:
        onnx_wheel_path = Path(onnx_wheel_path)
        if not onnx_wheel_path.exists():
            raise ValueError(f"onnx wheel file not found: {onnx_wheel_path}")
    
    # determine project root from experiment path if possible
    if 'experiments' in str(experiment_path):
        project_root = Path(str(experiment_path).split('experiments')[0])
    else:
        project_root = Path(os.getcwd())
    
    print(f"using project root: {project_root}")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = experiment_path.name
    zip_filename = f"{output_dir}/{experiment_name}_{timestamp}.zip"
    print(f"creating archive: {zip_filename}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # copy source directories
        for directory in ["src", "scripts", "shell_scripts"]:
            src_dir = project_root / directory
            if src_dir.exists():
                dst_dir = temp_path / directory
                shutil.copytree(src_dir, dst_dir)
                print(f"added directory: {directory}")
            else:
                print(f"warning: directory not found: {src_dir}")
        
        # create experiment folder for configs, plots, weights etc.
        exp_temp_path = temp_path / "experiment"
        os.makedirs(exp_temp_path, exist_ok=True)
        
        # copy experiment config files and plots
        for file in experiment_path.glob("*.json"):
            shutil.copy2(file, exp_temp_path)
            print(f"added config file: {file.name}")
        for file in experiment_path.glob("*.png"):
            shutil.copy2(file, exp_temp_path)
            print(f"added plot: {file.name}")
        
        # include only best weight file if available
        weights_dir = experiment_path / "weights"
        if weights_dir.exists():
            exp_weights_dir = exp_temp_path / "weights"
            os.makedirs(exp_weights_dir, exist_ok=True)
            
            best_weight_files = list(weights_dir.glob("best_*_step_*.pt"))
            if best_weight_files:
                def get_step_number(file_path):
                    match = re.search(r'step_(\d+)\.pt', str(file_path))
                    return int(match.group(1)) if match else 0
                best_weight_file = sorted(best_weight_files, key=get_step_number)[-1]
                shutil.copy2(best_weight_file, exp_weights_dir)
                print(f"added best weight file: {best_weight_file.name}")
            else:
                print("warning: no best weight files found!")
        
        # include pretrained model if provided
        if pretrained_path:
            pretrained_temp_path = temp_path / "pretrained"
            if pretrained_path.is_file():
                os.makedirs(pretrained_temp_path, exist_ok=True)
                shutil.copy2(pretrained_path, pretrained_temp_path)
                print(f"added pretrained model file: {pretrained_path.name}")
            elif pretrained_path.is_dir():
                shutil.copytree(pretrained_path, pretrained_temp_path)
                print(f"added pretrained model files from: {pretrained_path}")
        
        # create dependencies folder and download onnxruntime wheel if enabled
        deps_dir = temp_path / "dependencies"
        os.makedirs(deps_dir, exist_ok=True)
        
        if download_onnx:
            downloaded_wheel = download_onnxruntime(deps_dir)
            if downloaded_wheel:
                onnx_wheel_path = downloaded_wheel
        
        # create installation script for onnxruntime (using the local wheel)
        install_script = deps_dir / "install_dependencies.sh"
        with open(install_script, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# function to check if onnxruntime is installed\n")
            f.write("check_onnx() {\n")
            f.write("    python -c \"import onnxruntime\" 2>/dev/null\n")
            f.write("    return $?\n")
            f.write("}\n\n")
            f.write("SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n\n")
            f.write("if check_onnx; then\n")
            f.write("    echo 'onnxruntime is already installed'\n")
            f.write("else\n")
            f.write("    echo 'installing onnxruntime from local wheel...'\n")
            f.write("    WHEEL_FILE=$(find \"$SCRIPT_DIR\" -name \"onnxruntime*.whl\" | head -n 1)\n")
            f.write("    if [ -n \"$WHEEL_FILE\" ]; then\n")
            f.write("        echo \"installing from local wheel: $(basename \"$WHEEL_FILE\")\"\n")
            f.write("        pip install \"$WHEEL_FILE\"\n")
            f.write("    else\n")
            f.write("        echo 'no onnxruntime wheel found in dependencies'\n")
            f.write("    fi\n")
            f.write("    if check_onnx; then\n")
            f.write("        echo 'onnxruntime installation verified'\n")
            f.write("    else\n")
            f.write("        echo 'warning: onnxruntime installation failed'\n")
            f.write("    fi\n")
            f.write("fi\n")
        os.chmod(install_script, 0o755)
        print("added installation script: install_dependencies.sh")
        
        # create the zip archive
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
    
    print(f"archive created successfully: {zip_filename}")
    return zip_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export project with onnxruntime dependency for offline kaggle use")
    parser.add_argument("--onnx_wheel", type=str, help="local path to onnxruntime wheel to include")
    parser.add_argument("--download_onnx", action="store_true", help="download onnxruntime wheel during zipping")
    args, unknown = parser.parse_known_args()
    
    wheel_path = args.onnx_wheel if args.onnx_wheel else None
    should_download = args.download_onnx if args.download_onnx else download_onnx
    
    try:
        zip_path = create_zip_archive(experiment_path, output_dir, pretrained_path, wheel_path, should_download)
        print(f"\nexport complete! archive saved to: {zip_path}")
        print("\nto install dependencies on kaggle, add this to a notebook cell:")
        print("  !bash /kaggle/input/your-dataset-name/dependencies/install_dependencies.sh")
    except Exception as e:
        print(f"error creating export: {e}")
