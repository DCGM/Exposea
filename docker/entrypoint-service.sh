#!/bin/bash
set -e

# Path to the directory containing subfolders
INPUT_DIR="/data"
OUTPUT_DIR="/output"

while true; do
  FOLDERS=("$INPUT_DIR"/*/)
  echo $FOLDERS
  for dir in $FOLDERS; do
        [ -d "$dir" ] || continue
        base="$(basename "$dir")"

        if [[ "$base" != *_processing && "$base" != *_done ]]; then
            echo "Found: $base at $dir"

            PROCESS_DIR="$INPUT_DIR/$base"

            mv "$dir" "${PROCESS_DIR}_processing"

            echo "Processing: ${PROCESS_DIR}_processing"
            python3 register.py --output $OUTPUT_DIR --input "${PROCESS_DIR}_processing"

            mv "${PROCESS_DIR}_processing" "${PROCESS_DIR}_done"
            echo "Done: ${PROCESS_DIR}_done"
        fi
    done
    sleep 5
done
