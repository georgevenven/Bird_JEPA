#!/usr/bin/env bash
# exit on errors, undefined variables, and propagate errors in pipelines
set -euo pipefail

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# Navigate to project root
cd "${SCRIPT_DIR}/.."
PROJECT_ROOT=$(pwd)

# required parameters
AUDIO_DIR="${PROJECT_ROOT}/BirdCLEF/train_audio"
INPUT_DIR="${PROJECT_ROOT}/BirdCLEF"

# Spectrogram parameters
MULTI_THREAD=false                # use single-thread for diagnostic processing
STEP_SIZE=119                     # step size for spectrogram generation
NFFT=1024                         # number of fft points for spectrogram
NUM_FILES=100                     # small number for quick testing

# don't change
TEMP_DIR="${PROJECT_ROOT}/temp/diagnostic"
mkdir -p "$TEMP_DIR"
echo "Created diagnostic directory: $TEMP_DIR"

# Create directories for spectrograms
SPEC_DIR="$TEMP_DIR/specs"
mkdir -p "$SPEC_DIR"

# Generate a small set of spectrograms for diagnostic purposes
echo "Generating diagnostic spectrograms from $NUM_FILES audio files..."
python3 "${PROJECT_ROOT}/src/birdclef_spectrogram_generator.py" \
        --src_dir "$AUDIO_DIR" \
        --dst_dir "$SPEC_DIR" \
        --train_csv "$INPUT_DIR/train.csv" \
        --step_size "$STEP_SIZE" \
        --nfft "$NFFT" \
        --max_files "$NUM_FILES" \
        --random_subset \
        --single_threaded

# Extract the taxonomy file from the input directory
TAXONOMY_FILE="$INPUT_DIR/taxonomy.csv"
TRAIN_CSV="$INPUT_DIR/train.csv"

# Run analysis on the generated spectrograms
echo "Running label assignment analysis..."
python3 "${PROJECT_ROOT}/src/finetuning.py" \
    --mode analyze \
    --train_spec_dir "$SPEC_DIR" \
    --taxonomy_file "$TAXONOMY_FILE" \
    --train_csv "$TRAIN_CSV" \
    --analyze_samples 100

echo "Diagnostic analysis completed. Results shown above." 