#!/bin/bash

set -e

# Path to the directory containing subfolders
INPUT_DIR="/app/test_data"
OUTPUT_DIR="/app/output"
EXP_TAG=$1

# Get list of subfolders
FOLDERS=("$INPUT_DIR"/*/)

for dir in "${FOLDERS[@]}"; do
    [ -d "$dir" ] || continue
    base="$(basename "$dir")"

    if [[ "$base" != *_processing && "$base" != *_done && "$base" != *_incomplete ]]; then
        echo "Found: $base at $dir"

        PROCESS_DIR="$INPUT_DIR/$base"

        mv "$dir" "${PROCESS_DIR}_processing"

        echo "Processing: ${PROCESS_DIR}_processing"

        if [ -z "$EXP_TAG" ]; then
            # no EXP_TAG provided
            python3 /app/register.py \
                --output "$OUTPUT_DIR" \
                --input "${PROCESS_DIR}_processing"
        else
            # EXP_TAG provided
            python3 /app/register.py \
                --output "$OUTPUT_DIR" \
                --input "${PROCESS_DIR}_processing" \
                -e "$EXP_TAG"
        fi

        mv "${PROCESS_DIR}_processing" "${PROCESS_DIR}_done"
        echo "Done: ${PROCESS_DIR}_done"
    else
        echo "File $base is not eligible to process"
    fi
done
