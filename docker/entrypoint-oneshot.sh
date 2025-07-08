#!/bin/bash
set -e
# Path to the directory containing subfolders
INPUT_DIR="/data"
OUTPUT_DIR="/output"

FOLDERS=("$INPUT_DIR"/*/)

# Check if any directories exist
if [ ${#FOLDERS[@]} -eq 0 ]; then
    echo "No folders found in $INPUT_DIR"
    exit 1
fi

SEL_FOLDER="${FOLDERS[0]}"
echo "Processing $SEL_FOLDER"
echo "Saving to $OUTPUT_DIR"
# Check if contains config file and img files

python3 register.py --output $OUTPUT_DIR --input $SEL_FOLDER


