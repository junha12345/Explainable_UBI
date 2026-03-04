#!/usr/bin/env bash
set -x

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# Move to the project directory to ensure imports work
cd /workspace/VideoMAEv2

OUTPUT_DIR='/workspace/VideoMAEv2/work_dirs/custom_finetune'
DATA_PATH='/workspace/VideoMAEv2/data/custom'
MODEL_PATH='/workspace/VideoMAEv2/vit_b_k710_dl_from_giant.pth'

# Ensure output directory exists
mkdir -p $OUTPUT_DIR

# Run Fine-tuning
# Note: We use Custom dataset and nb_classes=2
# Batch size is set conservatively to 4 to avoid OOM on typical GPUs for VideoMAE
# If user has high VRAM, can increase.
# epochs 10 for quick turnaround, user can increase.
# frame_sampling_rate=4, num_frames=16 -> covers 16*4 = 64 frames. 
# Video is ~145 frames. 64 frames is ~2 seconds.
# We might want to sample more sparsely if we want to cover full 5 seconds (145 frames). 
# 145 / 16 ~ 9. So sampling_rate=9 would cover whole video.
# Standard VideoMAE is 16x4. Let's stick to standard 16 frames, but sampling rate?
# Let's use sampling_rate=8 (covers 128 frames ~ 4.2 seconds). Better coverage.

python3 run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set Custom \
    --nb_classes 2 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 4 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 5 \
    --num_frames 16 \
    --sampling_rate 8 \
    --num_sample 2 \
    --num_workers 4 \
    --opt adamw \
    --lr 5e-3 \
    --drop_path 0.1 \
    --clip_grad 5.0 \
    --layer_decay 0.75 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --warmup_epochs 2 \
    --epochs 20 \
    --dist_eval \
    --data_root ''
