#!/usr/bin/env bash
# exit on errors, undefined variables, and propagate errors in pipelines
set -euo pipefail

# Parameter to control whether to recreate temp and generate specs
recreate_temp=true  # set to false to skip temp/spec generation

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# Navigate to project root
cd "${SCRIPT_DIR}/.."
PROJECT_ROOT=$(pwd)

# Create experiment directory
EXPERIMENT_NAME="experiments/BirdJEPA_Small_Finetune"
PRETRAINED_MODEL_PATH="experiments/BirdJEPA_Small_Untrained"
DATA_PATH="BirdCLEF"

mkdir -p "$EXPERIMENT_NAME"
echo "Created experiment directory: $EXPERIMENT_NAME"

# Path variables
AUDIO_DIR="${DATA_PATH}/train_audio"
INPUT_DIR="${DATA_PATH}"
TRAIN_CSV="${INPUT_DIR}/train.csv"
TAXONOMY_FILE="${INPUT_DIR}/taxonomy.csv"

# Spectrogram parameters
STEP_SIZE=160
NFFT=1024
NUM_FILES=200  # Reduced file count for quick testing
MULTI_THREAD=true
SONG_DETECTION_JSON_PATH=None
TEST_PERCENTAGE=20

# # don't change
TEMP_DIR="./temp"
TRAIN_FILE_LIST="$TEMP_DIR/train_files.txt"
TEST_FILE_LIST="$TEMP_DIR/test_files.txt"
# # Define the experiment directory at the beginning

EXPERIMENT_DIR="$EXPERIMENT_NAME"
mkdir -p "$EXPERIMENT_DIR"
# echo "Created experiment directory: $EXPERIMENT_DIR"

# Define the spectrogram directories for the finetuning script
TRAIN_SPEC_DIR="$TEMP_DIR/train_dir"
VAL_SPEC_DIR="$TEMP_DIR/test_dir"

if [ "$recreate_temp" = true ]; then
    # remove the temp directory if it exists to avoid interference
    if [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
        echo "removed existing temporary directory: $TEMP_DIR"
    fi

    # create temp directory
    mkdir -p "$TEMP_DIR"
    echo "created temporary directory: $TEMP_DIR"

    # 1. split files into train and test (python writes train_files.txt and test_files.txt)
    python3 scripts/test_train_split.py "$AUDIO_DIR" "$TEST_PERCENTAGE" \
            --train_output "$TRAIN_FILE_LIST" \
            --test_output "$TEST_FILE_LIST" \
            --full_paths

    # 2. read the file lists into arrays
    train_files=()
    while IFS= read -r line; do
      train_files+=( "$line" )
    done < "$TRAIN_FILE_LIST"

    test_files=()
    while IFS= read -r line; do
      test_files+=( "$line" )
    done < "$TEST_FILE_LIST"

    # 3. print only counts (not all filenames)
    echo "found ${#train_files[@]} training files and ${#test_files[@]} testing files."

    # 4. create directories for wav files
    TRAIN_WAV_DIR="$TEMP_DIR/train_wav"
    TEST_WAV_DIR="$TEMP_DIR/test_wav"

    mkdir -p "$TRAIN_WAV_DIR"
    mkdir -p "$TEST_WAV_DIR"

    # 7. create train_dir and test_dir for spectrograms
    TRAIN_DIR="$TEMP_DIR/train_dir"
    TEST_DIR="$TEMP_DIR/test_dir"

    mkdir -p "$TRAIN_DIR"
    mkdir -p "$TEST_DIR"

    # determine number of processes for spectrogram generation
    # Use sysctl for macOS and fallback to logical approach for other systems
    PROCESS_COUNT=1
    if [ "$MULTI_THREAD" = true ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS uses sysctl
            PROCESS_COUNT=$(sysctl -n hw.ncpu 2>/dev/null || echo 2)
        else
            # Linux and others try nproc, if not available default to 2
            PROCESS_COUNT=$(nproc 2>/dev/null || echo 2)
        fi
        echo "Using $PROCESS_COUNT CPU cores for processing"
    fi

    # generate spectrograms (train + test)
    python3 src/spectrogram_generator.py \
            --file_list "$TRAIN_FILE_LIST" \
            --dst_dir "$TRAIN_DIR" \
            --song_detection_json_path "$SONG_DETECTION_JSON_PATH" \
            --step_size "$STEP_SIZE" \
            --nfft "$NFFT" \
            --single_threaded "$(if [[ "$MULTI_THREAD" == false ]]; then echo 'true'; else echo 'false'; fi)"

    python3 src/spectrogram_generator.py \
            --file_list "$TEST_FILE_LIST" \
            --dst_dir "$TEST_DIR" \
            --song_detection_json_path "$SONG_DETECTION_JSON_PATH" \
            --step_size "$STEP_SIZE" \
            --nfft "$NFFT" \
            --single_threaded "$(if [[ "$MULTI_THREAD" == false ]]; then echo 'true'; else echo 'false'; fi)"
fi

# Run baseline training, TRAIN CSV: this is for both train and val
echo "Running finetuning with pre-generated spectrograms..."
python3 "${PROJECT_ROOT}/src/finetuning.py" \
    --mode train \
    --train_spec_dir "$TRAIN_SPEC_DIR" \
    --val_spec_dir "$VAL_SPEC_DIR" \
    --train_csv "$TRAIN_CSV" \
    --output_dir "$EXPERIMENT_DIR" \
    --batch_size 36 \
    --max_steps 50000 \
    --save_interval 500 \
    --pretrained_model_path "$PRETRAINED_MODEL_PATH"
