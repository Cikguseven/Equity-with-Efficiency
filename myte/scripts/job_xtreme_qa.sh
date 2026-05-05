#!/bin/bash

MODEL_DIR="../models"
MODEL_TYPE="myt5"
MODEL_SIZE="large-qa_in_lang-v2"
TIMESTAMP="$(date +%b%d_%H%M)"
LOG_FILE="../logs/QA_${TIMESTAMP}_${MODEL_TYPE}_${MODEL_SIZE}.log"

mkdir -p ../logs

CUDA_VISIBLE_DEVICES=2 nohup python3.10 ../src/xtreme_qa.py \
    --model_dir $MODEL_DIR \
    --model_type $MODEL_TYPE \
    --model_size $MODEL_SIZE \
    > "$LOG_FILE" 2>&1 &
