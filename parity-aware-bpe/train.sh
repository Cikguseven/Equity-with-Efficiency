#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

LOG=/home/user/project/parity_aware_bpe/output_train.log
exec >"$LOG" 2>&1

START_EPOCH=$(date +%s)
echo "===== START: $(date -Is) ====="

on_exit () {
  rc=$?
  end_epoch=$(date +%s)
  echo "===== END:   $(date -Is) (rc=$rc, elapsed=$((end_epoch - START_EPOCH)) seconds) ====="
}
trap on_exit EXIT

/home/user/miniconda3/envs/project/bin/python \
/home/user/project/parity_aware_bpe/parity_aware_learn_bpe.py \
--symbols 90112 \
--variant "base" \
--output /home/user/project/parity_aware_bpe/90k_parity-aware_SEA_1m.out \
--num-workers 120 \
--input \
    /home/user/project/data/mc4_SEA_1000000_sentences_proportional/en.txt \
    /home/user/project/data/mc4_SEA_1000000_sentences_proportional/fil.txt \
    /home/user/project/data/mc4_SEA_1000000_sentences_proportional/id.txt \
    /home/user/project/data/mc4_SEA_1000000_sentences_proportional/km.txt \
    /home/user/project/data/mc4_SEA_1000000_sentences_proportional/lo.txt \
    /home/user/project/data/mc4_SEA_1000000_sentences_proportional/ms.txt \
    /home/user/project/data/mc4_SEA_1000000_sentences_proportional/my.txt \
    /home/user/project/data/mc4_SEA_1000000_sentences_proportional/ta.txt \
    /home/user/project/data/mc4_SEA_1000000_sentences_proportional/th.txt \
    /home/user/project/data/mc4_SEA_1000000_sentences_proportional/vi.txt \
    /home/user/project/data/mc4_SEA_1000000_sentences_proportional/zh.txt \
--dev \
    /home/user/project/data/flores-plus_dev_devtest/eng_Latn.dev \
    /home/user/project/data/flores-plus_dev_devtest/fil_Latn.dev \
    /home/user/project/data/flores-plus_dev_devtest/ind_Latn.dev \
    /home/user/project/data/flores-plus_dev_devtest/khm_Khmr.dev \
    /home/user/project/data/flores-plus_dev_devtest/lao_Laoo.dev \
    /home/user/project/data/flores-plus_dev_devtest/zsm_Latn.dev \
    /home/user/project/data/flores-plus_dev_devtest/mya_Mymr.dev \
    /home/user/project/data/flores-plus_dev_devtest/tam_Taml.dev \
    /home/user/project/data/flores-plus_dev_devtest/tha_Thai.dev \
    /home/user/project/data/flores-plus_dev_devtest/vie_Latn.dev \
    /home/user/project/data/flores-plus_dev_devtest/cmn_Hans.dev
