#!/bin/bash
# The script to download the MUSE dictionaries

set -euo pipefail
shopt -s nullglob

exec > /home/user/project/myte/output_download_muse_dict.txt 2>&1

SAVE_DIR="/home/user/project/myte/lexicon_dict"
mkdir -p "${SAVE_DIR}"

LANGUAGES=(af ar bg bn ca cs da de el en es et fa fi fr he hi hu id it ja ko lt lv mk ms nl no pl pt ro ru sk sl sq sv ta th tr uk vi zh)

for lang in "${LANGUAGES[@]}"
do
    wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-${lang}.txt -O ${SAVE_DIR}/${lang}_dict.txt
done
