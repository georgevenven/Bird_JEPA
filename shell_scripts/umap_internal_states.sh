#!/bin/bash

EXPERIMENT_FOLDER="experiments/bird_jepa_experiment"
DATA_DIRS=("/media/george-vengrovski/George-SSD/llb_stuff/llb3_test") # replace with actual dirs
SAVE_NAME="bird_jepa_experiment"

# Add the path to the src directory
cd "$(dirname "$0")/.." # Move up one directory from shell_scripts

python3 src/analysis.py \
    --experiment_folder "$EXPERIMENT_FOLDER" \
    --data_dirs "${DATA_DIRS[@]}" \
    --save_name "$SAVE_NAME" \
    --samples 10000 \
    --layer_index -2 \
    --dict_key "attention_output" \
    --context 500 \
    --min_cluster_size 500
