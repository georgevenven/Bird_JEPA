#!/usr/bin/env bash
# exit on errors, undefined variables, and propagate errors in pipelines
set -euo pipefail

# navigate up one directory
cd ..

# Mode selection (train or infer)
MODE=${1:-"train"}  # Default to train if no argument provided

# Python executable (update this to match your environment)
PYTHON_EXEC="/Users/georgev/anaconda3/envs/tweetybert/bin/python"

# Required parameters (change these for your setup)
DATA_PATH="/Users/georgev/Documents/data/birdclef-2025"  # Root directory of BirdCLEF dataset
TAXONOMY_FILE="$DATA_PATH/taxonomy.csv"  # Path to taxonomy file
TRAIN_CSV="$DATA_PATH/train.csv"  # Path to train.csv file
PRETRAINED_MODEL_PATH="experiments/BirdJEPA_Test"  # Path to pretrained model directory
OUTPUT_DIR="experiments/BirdCLEF_finetuned"  # Output directory for fine-tuned model

# For inference mode
MODEL_PATH="$OUTPUT_DIR/best_model.pt"  # Path to the fine-tuned model
AUDIO_PATH=""  # Path to the audio file or directory for inference
OUTPUT_FILE=""  # Path to save predictions for inference

# Optional parameters with defaults
VAL_PERCENTAGE=10.0
BATCH_SIZE=16
LEARNING_RATE=1e-4
EPOCHS=30
EARLY_STOPPING_PATIENCE=5
MAX_SAMPLES_PER_SPECIES=100
FREEZE_ENCODER=true
MULTI_THREAD=true
STEP_SIZE=119
NFFT=1024
SONG_DETECTION_JSON_PATH="None"  # Default to None
TEMP_DIR="$OUTPUT_DIR/temp_finetuning"

# For testing/debugging
DEBUG_MODE=true  # Set to false for full run
if [ "$DEBUG_MODE" = "true" ]; then
    echo "DEBUG MODE: Using reduced dataset for faster testing"
    MAX_FILES=100  # Limit number of files for debugging
    MAX_RANDOM_FILES=50  # Limit number of files for spectrogram generation
else
    MAX_FILES=""  # No limit
    MAX_RANDOM_FILES=""  # No limit
fi

# Check that the first argument is valid
if [ "$MODE" != "train" ] && [ "$MODE" != "infer" ]; then
    echo "Error: Mode must be 'train' or 'infer'"
    echo "Usage: ./finetune.sh [train|infer]"
    exit 1
fi

# Print configuration
echo "-------------------------"
echo "FINE-TUNING CONFIGURATION"
echo "-------------------------"
echo "Mode: $MODE"
echo "Data path: $DATA_PATH"
echo "Pretrained model: $PRETRAINED_MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Validation percentage: $VAL_PERCENTAGE%"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Step size: $STEP_SIZE"
echo "NFFT: $NFFT"
echo "-------------------------"

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "Created output directory: $OUTPUT_DIR"

# Function to run training
run_training() {
    # Build command
    CMD=(
        "$PYTHON_EXEC" src/finetuning.py
        --mode train
        --pretrained_model_path "$PRETRAINED_MODEL_PATH"
        --data_path "$DATA_PATH"
        --taxonomy_file "$TAXONOMY_FILE"
        --train_csv "$TRAIN_CSV"
        --output_dir "$OUTPUT_DIR"
        --val_percentage "$VAL_PERCENTAGE"
        --step_size "$STEP_SIZE"
        --nfft "$NFFT"
        --batch_size "$BATCH_SIZE"
        --learning_rate "$LEARNING_RATE"
        --epochs "$EPOCHS"
        --early_stopping_patience "$EARLY_STOPPING_PATIENCE"
        --max_samples_per_species "$MAX_SAMPLES_PER_SPECIES"
        --temp_dir "$TEMP_DIR"
    )

    # Add optional parameters
    if [ "$FREEZE_ENCODER" = true ]; then
        CMD+=(--freeze_encoder)
    fi

    if [ "$MULTI_THREAD" = true ]; then
        CMD+=(--multi_thread)
    fi

    # Add debug mode parameters if enabled
    if [ "$DEBUG_MODE" = true ]; then
        if [ -n "$MAX_FILES" ]; then
            CMD+=(--max_files "$MAX_FILES")
        fi
        if [ -n "$MAX_RANDOM_FILES" ]; then
            CMD+=(--max_random_files "$MAX_RANDOM_FILES")
        fi
    fi

    if [ "$SONG_DETECTION_JSON_PATH" != "None" ]; then
        CMD+=(--song_detection_json_path "$SONG_DETECTION_JSON_PATH")
    fi

    # Run command
    echo "Running fine-tuning..."
    echo "${CMD[@]}"
    "${CMD[@]}"

    echo "Fine-tuning complete! Model saved to $OUTPUT_DIR"
}

# Function to run inference
run_inference() {
    # Check that MODEL_PATH and AUDIO_PATH are set
    if [ -z "$MODEL_PATH" ] || [ -z "$AUDIO_PATH" ]; then
        echo "Error: MODEL_PATH and AUDIO_PATH must be set for inference mode"
        echo "Please edit the script to set these variables"
        exit 1
    fi

    # Build command
    CMD=(
        "$PYTHON_EXEC" src/finetuning.py
        --mode infer
        --model_path "$MODEL_PATH"
        --audio_path "$AUDIO_PATH"
        --taxonomy_file "$TAXONOMY_FILE"
        --step_size "$STEP_SIZE"
        --nfft "$NFFT"
        --temp_dir "$TEMP_DIR"
    )

    # Add optional output file
    if [ -n "$OUTPUT_FILE" ]; then
        CMD+=(--output_file "$OUTPUT_FILE")
    fi

    # Run command
    echo "Running inference..."
    echo "${CMD[@]}"
    "${CMD[@]}"

    echo "Inference complete!"
    if [ -n "$OUTPUT_FILE" ]; then
        echo "Predictions saved to $OUTPUT_FILE"
    fi
}

# Run the appropriate function based on mode
if [ "$MODE" = "train" ]; then
    run_training
else
    run_inference
fi
