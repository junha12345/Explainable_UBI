#!/bin/bash

# Directory containing test videos
VIDEO_DIR="../test_video"
OUTPUT_DIR="../test_video_vis"
CHECKPOINT="work_dirs/custom_finetune/checkpoint-19.pth"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Iterate over videos
for video in "$VIDEO_DIR"/*.mp4; do
    if [ -f "$video" ]; then
        echo "Processing $video..."
        python3 visualize_attention.py \
            --video_path "$video" \
            --checkpoint "$CHECKPOINT" \
            --output_dir "$OUTPUT_DIR" \
            --model vit_base_patch16_224 \
            --num_frames 16
    fi
done

echo "All Done! Check outputs in $OUTPUT_DIR"
