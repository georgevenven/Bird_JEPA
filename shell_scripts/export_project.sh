#!/bin/bash

EXPERIMENT_PATH="/home/george-vengrovski/Documents/projects/Bird_JEPA/experiments/BirdJEPA_Run_10k_Finetune"


# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Get parameters
OUTPUT_DIR="${2:-zips}"  # Default to "zips" if not provided

# Create output directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/$OUTPUT_DIR"

# Execute the Python script
cd "$PROJECT_ROOT"
python3 scripts/export_project.py "$EXPERIMENT_PATH" --output-dir "$OUTPUT_DIR"

# Make script executable
chmod +x "$SCRIPT_DIR/export_project.sh"

echo "Done!" 