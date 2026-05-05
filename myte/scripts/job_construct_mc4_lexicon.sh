#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

LANGUAGES=('af' 'am' 'ar' 'az' 'be' 'bg' 'bn' 'ca' 'ceb' 'co' 'cs' 'cy' 'da' 'de' 'el' 'en' 'eo' 'es' 'et' 'eu' 'fa' 'fi' 'fo' 'fr' 'ga' 'gd' 'gl' 'gu' 'ha' 'haw' 'he' 'hi' 'ht' 'hu' 'hy' 'id' 'ig' 'is' 'it' 'ja' 'jv' 'ka' 'kk' 'km' 'kn' 'ko' 'ku' 'ky' 'la' 'lb' 'lo' 'lt' 'lv' 'mg' 'mi' 'mk' 'ml' 'mn' 'mr' 'ms' 'mt' 'my' 'ne' 'nl' 'no' 'ny' 'pa' 'pl' 'ps' 'pt' 'ro' 'ru' 'sd' 'si' 'sk' 'sl' 'sm' 'sn' 'so' 'sq' 'sr' 'st' 'su' 'sv' 'sw' 'ta' 'te' 'tg' 'th' 'tr' 'uk' 'ur' 'uz' 'vi' 'xh' 'yi' 'yo' 'zh' 'zu')
LANGUAGES=('en' 'fil' 'id' 'km' 'lo' 'ms' 'my' 'ta' 'th' 'vi' 'zh')

LEXICON_SIZE=30000
SUFFIX="equal"
LEXICON_DIR=/home/user/project/myte/lexicon_mc4_${LEXICON_SIZE}_${SUFFIX}
MORPH_TYP_NUM=8192
CORPUS="mc4"
MODEL_DIR="/home/user/project/myte/morfessor_models_decomposed_filtered_${CORPUS}_30k_${SUFFIX}"

mkdir -p ${LEXICON_DIR}
mkdir -p ${MODEL_DIR}

START_EPOCH=$(date +%s)
echo "===== START: $(date -Is) ====="

for LANG in "${LANGUAGES[@]}"; do
  (
    echo "Creating lexicons for ${LANG}"

    /home/user/miniconda3/envs/project/bin/python \
    /home/user/project/myte/src/construct_mc4_lexicon.py \
    --language "$LANG" \
    --lexicon_directory /home/user/project/myte/muse_lexicon_dict \
    --output_directory ${LEXICON_DIR} \
    --corpus_path /home/user/project/data/mc4_SEA_1000000_sentences_equal \
    --pre_processing_file /home/user/project/myte/byte_maps/decompose.json \
    --do_capitalize \
    --min_occurrences 0 \
    --lexicon_size ${LEXICON_SIZE} \
    --filter_en
  ) &
done

wait

# Loop through languages and train morfessor in parallel, firing off background jobs with '&'
for LANG in "${LANGUAGES[@]}"; do
  (
    LEXICON_FILE="${LEXICON_DIR}/${LANG}_lex.txt"
    OUTPUT_MODEL="${MODEL_DIR}/${LANG}_${MORPH_TYP_NUM}.bin"

    echo "Training morfessor for ${LANG} aiming for ${MORPH_TYP_NUM} morph types..."

    if [[ "$MORPH_TYP_NUM" = 0 ]]; then
      morfessor-train --encoding "utf-8" --traindata-list "${LEXICON_FILE}" \
        -s "${OUTPUT_MODEL}" --atom-separator ' ' --max-epochs 50
    else
      morfessor-train --encoding "utf-8" --traindata-list "${LEXICON_FILE}" \
        -s "${OUTPUT_MODEL}" --atom-separator ' ' --max-epochs 50 \
        --num-morph-types "${MORPH_TYP_NUM}"
    fi

    echo "Finished ${LANG}"
  ) &
done


# Pauses the main script until all background jobs are finished.
echo "Waiting for all training jobs to complete..."
wait
echo "All training jobs finished. Starting Python map construction..."

# Run the Python script with required arguments
/home/user/miniconda3/envs/project/bin/python \
  /home/user/project/myte/src/construct_morf_map.py \
  --languages "${LANGUAGES[@]}" \
  --model_dir "${MODEL_DIR}" \
  --mapping_dir /home/user/project/myte/mappings_decomposed_filtered_${SUFFIX} \
  --suffix "_${CORPUS}_${MORPH_TYP_NUM}_${SUFFIX}" \
  --mtn "${MORPH_TYP_NUM}" \
  --sort_by_cost \
  --cluster_scripts \
  --method morfessor

end_epoch=$(date +%s)
echo "===== END:   $(date -Is) (elapsed=$((end_epoch - START_EPOCH)) seconds) ====="
