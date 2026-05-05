#!/bin/bash

MODEL_DIR="../models"
MODEL_TYPE="myt5"
MODEL_SIZE="large-translation-id-zh"
TIMESTAMP="$(date +%b%d_%H%M)"
LOG_FILE="../logs/MT_${TIMESTAMP}_${MODEL_TYPE}_${MODEL_SIZE}.log"

mkdir -p ../logs

CUDA_VISIBLE_DEVICES=1 nohup python3.10 ../src/xtreme_mt.py \
    --model_dir $MODEL_DIR \
    --model_type $MODEL_TYPE \
    --model_size $MODEL_SIZE \
    > "$LOG_FILE" 2>&1 &
