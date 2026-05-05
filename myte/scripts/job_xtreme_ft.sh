#!/bin/bash

MODEL_DIR="../models"
MODEL_TYPE="myt5"
MODEL_SIZE="large"
TASK="translation"
MODEL_NAME="id-zh"
MODEL_STEPS=250000
TIMESTAMP="$(date +%b%d_%H%M)"
LOG_FILE="../logs/FT_${TIMESTAMP}_${MODEL_SIZE}_${TASK}_${MODEL_NAME}.log"

mkdir -p ../logs

CUDA_VISIBLE_DEVICES=3 nohup python3.10 ../src/xtreme_ft.py \
    --model_dir $MODEL_DIR \
    --task $TASK \
    --model_type $MODEL_TYPE \
    --model_size $MODEL_SIZE \
    --model_name $MODEL_NAME \
    --model_steps $MODEL_STEPS \
    > "$LOG_FILE" 2>&1 &