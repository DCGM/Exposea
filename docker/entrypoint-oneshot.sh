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
# Check if contains config file and img files
if [ -f "$INPUT_DIR/$SEL_FOLDER/config.yaml" ] && [ -d "$INPUT_DIR/$SEL_FOLDER/images"]; then
  python3 register.py \
  --output $OUTPUT_DIR \
  --input $INPUT_DIR/$SEL_FOLDER
else
   echo "Could not find config file or image folder at $INPUT_DIR/$SEL_FOLDER/config.yaml"
fi



