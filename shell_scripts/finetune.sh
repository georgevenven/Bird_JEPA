#!/usr/bin/env bash
# exit on errors, undefined variables, and propagate errors in pipelines
set -euo pipefail

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# Navigate to project root
cd "${SCRIPT_DIR}/.."
PROJECT_ROOT=$(pwd)

# Create experiment directory
EXPERIMENT_DIR="${PROJECT_ROOT}/experiments/BirdJEPA_Run_10k_Finetune"
mkdir -p "$EXPERIMENT_DIR"
echo "Created experiment directory: $EXPERIMENT_DIR"

# Path variables
AUDIO_DIR="${PROJECT_ROOT}/BirdCLEF/train_audio"
INPUT_DIR="${PROJECT_ROOT}/BirdCLEF"
TRAIN_CSV="${INPUT_DIR}/train.csv"
TAXONOMY_FILE="${INPUT_DIR}/taxonomy.csv"

# Spectrogram parameters
STEP_SIZE=119
NFFT=1024
NUM_FILES=200  # Reduced file count for quick testing

# Create directories for spectrograms
TRAIN_SPEC_DIR="${PROJECT_ROOT}/temp/train_dir"
VAL_SPEC_DIR="${PROJECT_ROOT}/temp/test_dir"
mkdir -p "$TRAIN_SPEC_DIR"
mkdir -p "$VAL_SPEC_DIR"

# Generate spectrograms using the wrapper script
echo "Generating spectrograms from audio files in $AUDIO_DIR"

# Generate training spectrograms
echo "Generating training spectrograms..."
python3 "${PROJECT_ROOT}/src/birdclef_wrapper.py" \
        --src_dir "$AUDIO_DIR" \
        --dst_dir "$TRAIN_SPEC_DIR" \
        --train_csv "$TRAIN_CSV" \
        --step_size "$STEP_SIZE" \
        --nfft "$NFFT" \
        --max_files "$NUM_FILES" \
        --random_subset

# Generate validation spectrograms with a smaller subset
echo "Generating validation spectrograms with 50 files..."
python3 "${PROJECT_ROOT}/src/birdclef_wrapper.py" \
        --src_dir "$AUDIO_DIR" \
        --dst_dir "$VAL_SPEC_DIR" \
        --train_csv "$TRAIN_CSV" \
        --step_size "$STEP_SIZE" \
        --nfft "$NFFT" \
        --max_files 50 \
        --random_subset

# Run baseline training
echo "Running baseline training with pre-generated spectrograms..."
python3 "${PROJECT_ROOT}/src/finetuning.py" \
    --mode train \
    --train_spec_dir "$TRAIN_SPEC_DIR" \
    --val_spec_dir "$VAL_SPEC_DIR" \
    --taxonomy_file "$TAXONOMY_FILE" \
    --train_csv "$TRAIN_CSV" \
    --output_dir "$EXPERIMENT_DIR" \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --max_steps 100 \
    --early_stopping_patience 10 \
    --save_interval 10 \
    --eval_interval 5 \
    --pretrained_model_path "/home/george-vengrovski/Documents/projects/Bird_JEPA/experiments/BirdJEPA_Run_10k/saved_weights/checkpoint_9999.pt"