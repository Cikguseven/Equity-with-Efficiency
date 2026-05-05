#!/bin/bash

# Define paths and parameters for downloading corpus
DECOMPOSE_MAP="../byte_maps/decompose_map.json"
MERGE_MAP="../byte_maps/merge_map.json"
MAX_ARTICLES=100000 # Adjust as needed
CORPUS_BYTE_OUTPUT_DIR="../bpe_corpus"  # Output path for the corpus

# Define log file
TIMESTAMP="$(date +%b%d_%H%M)"
LOG_FILE="../logs/CORPUS_DL_${MAX_ARTICLES}_${TIMESTAMP}.log"

# Create directories if they don't exist
mkdir -p ../logs
mkdir -p "$CORPUS_BYTE_OUTPUT_DIR"

# Run the pretrain_BPE.py script using nohup for background execution
nohup python3.10 ../src/pretrain_download_corpus.py \
    --decompose_map $DECOMPOSE_MAP \
    --merge_map $MERGE_MAP \
    --max_articles $MAX_ARTICLES \
    --corpus_byte_output_dir $CORPUS_BYTE_OUTPUT_DIR \
    > $LOG_FILE 2>&1 &

echo "Corpus downloading script started in the background."
echo "Output and errors will be logged to: $LOG_FILE"
echo "Corpus will be saved to: $CORPUS_BYTE_OUTPUT_DIR"