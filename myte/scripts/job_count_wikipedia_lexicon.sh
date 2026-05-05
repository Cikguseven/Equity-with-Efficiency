#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

LOG=/home/user/project/myte/output_count_wikipedia_lexicon.txt
exec > "$LOG" 2>&1

START_EPOCH=$(date +%s)
echo "===== START: $(date -Is) ====="

on_exit () {
  rc=$?
  end_epoch=$(date +%s)
  echo "===== END:   $(date -Is) (rc=$rc, elapsed=$((end_epoch - START_EPOCH)) seconds) ====="
}
trap on_exit EXIT

LANGUAGES=('af' 'am' 'ar' 'az' 'be' 'bg' 'bn' 'ca' 'ceb' 'co' 'cs' 'cy' 'da' 'de' 'el' 'en' 'eo' 'es' 'et' 'eu' 'fa' 'fi' 'fo' 'fr' 'ga' 'gd' 'gl' 'gu' 'ha' 'haw' 'he' 'hi' 'ht' 'hu' 'hy' 'id' 'ig' 'is' 'it' 'ja' 'jv' 'ka' 'kk' 'km' 'kn' 'ko' 'ku' 'ky' 'la' 'lb' 'lo' 'lt' 'lv' 'mg' 'mi' 'mk' 'ml' 'mn' 'mr' 'ms' 'mt' 'my' 'ne' 'nl' 'no' 'ny' 'pa' 'pl' 'ps' 'pt' 'ro' 'ru' 'sd' 'si' 'sk' 'sl' 'sm' 'sn' 'so' 'sq' 'sr' 'st' 'su' 'sv' 'sw' 'ta' 'te' 'tg' 'th' 'tr' 'uk' 'ur' 'uz' 'vi' 'xh' 'yi' 'yo' 'zh' 'zu')

OUTPUT_DIR=/home/user/project/myte/lexicon_wiki
mkdir -p ${OUTPUT_DIR}

for LANG in "${LANGUAGES[@]}"; do
    echo "Creating wikipedia corpus for ${LANG}"

    /home/user/miniconda3/envs/project/bin/python \
    /home/user/project/myte/src/count_wikipedia_lexicon.py \
    --language "$LANG" \
    --trust_remote_code
done