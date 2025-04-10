#!/bin/bash

EXPERIMENT_FOLDER="experiments/BirdJEPA_LLB3_Large_Test"
DATA_DIRS=("/media/george-vengrovski/Desk SSD/TweetyBERT/linear_probe_dataset/llb3_test") # replace with actual dirs
SAVE_NAME="BirdJEPA_Run_10k"

# Add the path to the src directory
cd "$(dirname "$0")/.." # Move up one directory from shell_scripts

python3 src/analysis.py \
    --experiment_folder "$EXPERIMENT_FOLDER" \
    --data_dirs "${DATA_DIRS[@]}" \
    --save_name "$SAVE_NAME" \
    --samples 10000 \
    --layer_index -1 \
    --dict_key "attention_output" \
    --context 1000 \
    --min_cluster_size 500
