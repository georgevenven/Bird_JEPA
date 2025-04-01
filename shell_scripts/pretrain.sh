#!/usr/bin/env bash
# exit on errors, undefined variables, and propagate errors in pipelines
set -euo pipefail

# navigate up one directory
cd ..

# required parameters
INPUT_DIR="/Users/georgev/Documents/codebases/BirdJEPA/xeno_mp3s"
SONG_DETECTION_JSON_PATH=None
TEST_PERCENTAGE=20
EXPERIMENT_NAME="BirdJEPA_Test"

# change default parameters (if needed)
BATCH_SIZE=12                    # training batch size
LEARNING_RATE=3e-4                # learning rate for training
MULTI_THREAD=true                 # set to false for single-thread spectrogram generation
STEP_SIZE=119                     # step size for spectrogram generation
NFFT=1024                         # number of fft points for spectrogram
MAX_STEPS=5                   # maximum training steps
EVAL_INTERVAL=1                   # evaluation interval
INPUT_DIM=513                     # input dimension (frequency bins)
HIDDEN_DIM=64                    # hidden dimension
MAX_SEQ_LEN=500                   # maximum sequence length
MASK_RATIO=0.3                    # mask ratio for training
DEBUG=true                       # debug mode flag (lowercase)
# If debug mode is enabled, inform the user
if [ "$DEBUG" = "true" ]; then
    echo "Debug mode enabled - detailed logs will be saved to experiments/$EXPERIMENT_NAME/debug_log.txt"
fi

# New flexible architecture configuration
# Format: "type:param,type:param,..." where type is "local" or "global"
# For local blocks, param is window size
# For global blocks, param is stride
ARCHITECTURE="local:8,global:100,local:16,global:50"

# Import statement for block classes (add to trainer.py)
python_import="from model import LocalAttentionBlock, GlobalAttentionBlock"

# don't change
TEMP_DIR="./temp"
TRAIN_FILE_LIST="$TEMP_DIR/train_files.txt"
TEST_FILE_LIST="$TEMP_DIR/test_files.txt"
# Define the experiment directory at the beginning
EXPERIMENT_DIR="experiments/$EXPERIMENT_NAME"
mkdir -p "$EXPERIMENT_DIR"
echo "Created experiment directory: $EXPERIMENT_DIR"

# remove the temp directory if it exists to avoid interference
if [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
    echo "removed existing temporary directory: $TEMP_DIR"
fi

# create temp directory
mkdir -p "$TEMP_DIR"
echo "created temporary directory: $TEMP_DIR"

# 1. split files into train and test (python writes train_files.txt and test_files.txt)
python3 scripts/test_train_split.py "$INPUT_DIR" "$TEST_PERCENTAGE" \
        --train_output "$TRAIN_FILE_LIST" \
        --test_output "$TEST_FILE_LIST"

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

# 5. copy train files into TRAIN_WAV_DIR (flat structure)
for file in "${train_files[@]}"; do
    base=$(basename "$file")
    # search recursively for the file in the input directory
    src_path=$(find "$INPUT_DIR" -type f -name "$base" -print -quit)
    if [ -z "$src_path" ]; then
         echo "warning: file $base not found in $INPUT_DIR" >&2
         continue
    fi
    cp "$src_path" "$TRAIN_WAV_DIR/"
done

# 6. similarly, copy test files into TEST_WAV_DIR (flat structure)
for file in "${test_files[@]}"; do
    base=$(basename "$file")
    src_path=$(find "$INPUT_DIR" -type f -name "$base" -print -quit)
    if [ -z "$src_path" ]; then
         echo "warning: file $base not found in $INPUT_DIR" >&2
         continue
    fi
    cp "$src_path" "$TEST_WAV_DIR/"
done

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
        --src_dir "$TRAIN_WAV_DIR" \
        --dst_dir "$TRAIN_DIR" \
        --song_detection_json_path "$SONG_DETECTION_JSON_PATH" \
        --step_size "$STEP_SIZE" \
        --nfft "$NFFT" \
        --single_threaded "$([[ "$MULTI_THREAD" == false ]] && echo 'true' || echo 'false')" \
        --generate_random_files_number 10

        
python3 src/spectrogram_generator.py \
        --src_dir "$TEST_WAV_DIR" \
        --dst_dir "$TEST_DIR" \
        --song_detection_json_path "$SONG_DETECTION_JSON_PATH" \
        --step_size "$STEP_SIZE" \
        --nfft "$NFFT" \
        --single_threaded "$([[ "$MULTI_THREAD" == false ]] && echo 'true' || echo 'false')" \
        --generate_random_files_number 10


# run trainer with BirdJEPA model
# Pass the debug flag conditionally
if [ "$DEBUG" = "true" ]; then
    python3 src/trainer.py \
        --name "$EXPERIMENT_NAME" \
        --train_dir "$TRAIN_DIR" \
        --test_dir "$TEST_DIR" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --steps "$MAX_STEPS" \
        --eval_interval "$EVAL_INTERVAL" \
        --input_dim "$INPUT_DIM" \
        --hidden_dim "$HIDDEN_DIM" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --mask_ratio "$MASK_RATIO" \
        --architecture "$ARCHITECTURE" \
        --debug
else
    python3 src/trainer.py \
        --name "$EXPERIMENT_NAME" \
        --train_dir "$TRAIN_DIR" \
        --test_dir "$TEST_DIR" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --steps "$MAX_STEPS" \
        --eval_interval "$EVAL_INTERVAL" \
        --input_dim "$INPUT_DIM" \
        --hidden_dim "$HIDDEN_DIM" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --mask_ratio "$MASK_RATIO" \
        --architecture "$ARCHITECTURE"
fi

# 10. save file lists into the experiment folder
cp "$TRAIN_FILE_LIST" "$EXPERIMENT_DIR/train_files.txt"
cp "$TEST_FILE_LIST" "$EXPERIMENT_DIR/test_files.txt"
echo "copied train and test file lists to: $EXPERIMENT_DIR"

# 11. clean up temp directory
rm -rf "$TEMP_DIR"
echo "deleted temporary directory and its contents: $TEMP_DIR"