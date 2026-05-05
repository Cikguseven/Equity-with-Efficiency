#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

LOG=/home/user/project/parity_aware_bpe/output_create_tokenizer.log
exec > "$LOG" 2>&1

START_EPOCH=$(date +%s)
echo "===== START: $(date -Is) ====="

on_exit () {
  rc=$?
  end_epoch=$(date +%s)
  echo "===== END:   $(date -Is) (rc=$rc, elapsed=$((end_epoch - START_EPOCH)) seconds) ====="
}
trap on_exit EXIT

/home/user/miniconda3/envs/project/bin/python \
/home/user/project/parity_aware_bpe/HF_tokenizer.py \
--merges_file_path /home/user/project/parity_aware_bpe/90k_byte-level_SEA_1m_equal.out \
--tokenizer_path /home/user/project/parity_aware_bpe/90k_byte-level_SEA_1m_equal \