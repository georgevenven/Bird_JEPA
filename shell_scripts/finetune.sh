#!/usr/bin/env bash
# exit on errors, undefined variables, and propagate errors in pipelines
set -euo pipefail

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
# TRAIN_FILE_LIST="$TEMP_DIR/train_files.txt"
# TEST_FILE_LIST="$TEMP_DIR/test_files.txt"
# # Define the experiment directory at the beginning

EXPERIMENT_DIR="$EXPERIMENT_NAME"
# mkdir -p "$EXPERIMENT_DIR"
# echo "Created experiment directory: $EXPERIMENT_DIR"

# Define the spectrogram directories for the finetuning script
TRAIN_SPEC_DIR="$TEMP_DIR/train_dir"
VAL_SPEC_DIR="$TEMP_DIR/test_dir"

# # remove the temp directory if it exists to avoid interference
# if [ -d "$TEMP_DIR" ]; then
#     rm -rf "$TEMP_DIR"
#     echo "removed existing temporary directory: $TEMP_DIR"
# fi

# # create temp directory
# mkdir -p "$TEMP_DIR"
# echo "created temporary directory: $TEMP_DIR"

# # 1. split files into train and test (python writes train_files.txt and test_files.txt)
# python3 scripts/test_train_split.py "$AUDIO_DIR" "$TEST_PERCENTAGE" \
#         --train_output "$TRAIN_FILE_LIST" \
#         --test_output "$TEST_FILE_LIST"

# # 2. read the file lists into arrays
# train_files=()
# while IFS= read -r line; do
#   train_files+=( "$line" )
# done < "$TRAIN_FILE_LIST"

# test_files=()
# while IFS= read -r line; do
#   test_files+=( "$line" )
# done < "$TEST_FILE_LIST"

# # 3. print only counts (not all filenames)
# echo "found ${#train_files[@]} training files and ${#test_files[@]} testing files."

# # 4. create directories for wav files
# TRAIN_WAV_DIR="$TEMP_DIR/train_wav"
# TEST_WAV_DIR="$TEMP_DIR/test_wav"

# mkdir -p "$TRAIN_WAV_DIR"
# mkdir -p "$TEST_WAV_DIR"

# # 5. copy train files into TRAIN_WAV_DIR (flat structure)
# for file in "${train_files[@]}"; do
#     base=$(basename "$file")
#     # search recursively for the file in the input directory
#     src_path=$(find "$INPUT_DIR" -type f -name "$base" -print -quit)
#     if [ -z "$src_path" ]; then
#          echo "warning: file $base not found in $INPUT_DIR" >&2
#          continue
#     fi
#     cp "$src_path" "$TRAIN_WAV_DIR/"
# done

# # 6. similarly, copy test files into TEST_WAV_DIR (flat structure)
# for file in "${test_files[@]}"; do
#     base=$(basename "$file")
#     src_path=$(find "$INPUT_DIR" -type f -name "$base" -print -quit)
#     if [ -z "$src_path" ]; then
#          echo "warning: file $base not found in $INPUT_DIR" >&2
#          continue
#     fi
#     cp "$src_path" "$TEST_WAV_DIR/"
# done

# # 7. create train_dir and test_dir for spectrograms
# TRAIN_DIR="$TEMP_DIR/train_dir"
# TEST_DIR="$TEMP_DIR/test_dir"

# mkdir -p "$TRAIN_DIR"
# mkdir -p "$TEST_DIR"

# # determine number of processes for spectrogram generation
# # Use sysctl for macOS and fallback to logical approach for other systems
# PROCESS_COUNT=1
# if [ "$MULTI_THREAD" = true ]; then
#     if [[ "$OSTYPE" == "darwin"* ]]; then
#         # macOS uses sysctl
#         PROCESS_COUNT=$(sysctl -n hw.ncpu 2>/dev/null || echo 2)
#     else
#         # Linux and others try nproc, if not available default to 2
#         PROCESS_COUNT=$(nproc 2>/dev/null || echo 2)
#     fi
#     echo "Using $PROCESS_COUNT CPU cores for processing"
# fi

# # generate spectrograms (train + test)
# python3 src/spectrogram_generator.py \
#         --src_dir "$TRAIN_WAV_DIR" \
#         --dst_dir "$TRAIN_DIR" \
#         --song_detection_json_path "$SONG_DETECTION_JSON_PATH" \
#         --step_size "$STEP_SIZE" \
#         --nfft "$NFFT" \
#         --single_threaded "$(if [[ "$MULTI_THREAD" == false ]]; then echo 'true'; else echo 'false'; fi)"

        
# python3 src/spectrogram_generator.py \
#         --src_dir "$TEST_WAV_DIR" \
#         --dst_dir "$TEST_DIR" \
#         --song_detection_json_path "$SONG_DETECTION_JSON_PATH" \
#         --step_size "$STEP_SIZE" \
#         --nfft "$NFFT" \
#         --single_threaded "$(if [[ "$MULTI_THREAD" == false ]]; then echo 'true'; else echo 'false'; fi)"

# Run baseline training, TRAIN CSV: this is for both train and val
echo "Running finetuning with pre-generated spectrograms..."
python3 "${PROJECT_ROOT}/src/finetuning.py" \
    --mode infer \
    --train_spec_dir "$TRAIN_SPEC_DIR" \
    --val_spec_dir "$VAL_SPEC_DIR" \
    --taxonomy_file "$TAXONOMY_FILE" \
    --train_csv "$TRAIN_CSV" \
    --output_dir "$EXPERIMENT_DIR" \
    --batch_size 20 \
    --learning_rate 5e-4 \
    --max_steps 25000 \
    --early_stopping_patience 50 \
    --save_interval 1000 \
    --eval_interval 250 \
    --pretrained_model_path "$PRETRAINED_MODEL_PATH"
