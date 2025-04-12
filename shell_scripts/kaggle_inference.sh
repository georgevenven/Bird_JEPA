#!/usr/bin/env bash
set -euo pipefail

# debugging info: print basic environment variables
echo "=== DEBUG: starting script ==="
echo "BASH_SOURCE: ${BASH_SOURCE[0]}"
echo "initial pwd: $(pwd)"

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"

# navigate to project root (assumed to be one directory above the script directory)
cd "${SCRIPT_DIR}/.."
PROJECT_ROOT=$(pwd)
echo "PROJECT_ROOT (for info only): ${PROJECT_ROOT}"

# Parse command line arguments
EXPERIMENT_NAME=${1:-"Experiment_Path"}
DATA_PATH=${2:-"Data_Path"}
TRAIN_CSV=${3:-"Train_CSV"}
PRETRAINED_MODEL_PATH=${4:-"Pretrained_Model_Path"}

echo "EXPERIMENT_NAME: ${EXPERIMENT_NAME}"
echo "DATA_PATH: ${DATA_PATH}"
echo "TRAIN_CSV: ${TRAIN_CSV}"
echo "PRETRAINED_MODEL_PATH: ${PRETRAINED_MODEL_PATH}"

# Spectrogram parameters
STEP_SIZE=160
NFFT=1024
MULTI_THREAD=true
SONG_DETECTION_JSON_PATH=None

# Temp dir
TEMP_DIR="./temp"

# remove the temp directory if it exists to avoid interference
if [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
    echo "removed existing temporary directory: $TEMP_DIR"
fi

# create temp directory
mkdir -p "$TEMP_DIR"
echo "created temporary directory: $TEMP_DIR"

# spec dir
SPEC_DIR="$TEMP_DIR/spec"
mkdir -p "$SPEC_DIR"
echo "Created spectrogram directory: $SPEC_DIR"

# determine number of processes for spectrogram generation
PROCESS_COUNT=1
if [ "$MULTI_THREAD" = true ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        PROCESS_COUNT=$(sysctl -n hw.ncpu 2>/dev/null || echo 2)
    else
        PROCESS_COUNT=$(nproc 2>/dev/null || echo 2)
    fi
    echo "Using $PROCESS_COUNT CPU cores for processing"
fi

# generate spectrograms (train + test)
python3 src/spectrogram_generator.py \
    --src_dir "$DATA_PATH" \
    --dst_dir "$SPEC_DIR" \
    --song_detection_json_path "$SONG_DETECTION_JSON_PATH" \
    --step_size "$STEP_SIZE" \
    --nfft "$NFFT" \
    --single_threaded "$(if [[ "$MULTI_THREAD" == false ]]; then echo 'true'; else echo 'false'; fi)"

echo "Spectrogram generation completed."

# Run baseline training/inference using the generated spectrograms
echo "Running finetuning with pre-generated spectrograms..."
python3 src/finetuning.py \
    --mode infer \
    --train_spec_dir "$SPEC_DIR" \
    --val_spec_dir "$SPEC_DIR" \
    --train_csv "$TRAIN_CSV" \
    --batch_size 12 \
    --learning_rate 3e-4 \
    --max_steps 25000 \
    --early_stopping_patience 50 \
    --save_interval 1000 \
    --eval_interval 250 \
    --output_dir "$EXPERIMENT_NAME" \
    --pretrained_model_path "$PRETRAINED_MODEL_PATH"

echo "=== DEBUG: script completed ==="
