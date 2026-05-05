#!/bin/bash

# This script is used to run Morfessor on a corpus of text.

LANG_NAME=$1
SPLIT=$2
BYTE_PATCHES=$3
MORPH_LENGTH=$4
MORPH_TYP_NUM=$5
IF_JOINT=$6


DIR_NAME="/home/user/project/data/flores-200/${SPLIT}"
FILE_NAME="${DIR_NAME}/${LANG_NAME}.${SPLIT}"



LEXICON_DIR="/home/user/project/myte/morfessor_in"
EXPERIMENT_NAME="bp_${BYTE_PATCHES}_ml_${MORPH_LENGTH}_mtn_${MORPH_TYP_NUM}"

CORPUS_FILE="${LEXICON_DIR}/${LANG_NAME}/corpus_${SPLIT}.txt"

mkdir -p "${LEXICON_DIR}/${LANG_NAME}"

OUTPUT_DIR="/home/user/project/myte/morfessor_out"

if [ "$IF_JOINT" = 0 ]; then
  OUTPUT_MODEL="${OUTPUT_DIR}/${LANG_NAME}/model_${EXPERIMENT_NAME}.bin"
  OUTPUT_FILE="${OUTPUT_DIR}/${LANG_NAME}/${SPLIT}_${EXPERIMENT_NAME}.seg"
else
  OUTPUT_MODEL="${OUTPUT_DIR}/joint_model_${EXPERIMENT_NAME}.bin"
  OUTPUT_FILE="${OUTPUT_DIR}/${LANG_NAME}/joint_${SPLIT}_${EXPERIMENT_NAME}.seg"
fi

mkdir -p "${OUTPUT_DIR}/${LANG_NAME}"

# Prepare the lexicon for morfessor.
if [ "$BYTE_PATCHES" = 0 ] ; then
    /home/user/miniconda3/envs/project/bin/python /home/user/project/myte/src/prepare_corpus_lexicon.py "${FILE_NAME}" "${CORPUS_FILE}"
else
    /home/user/miniconda3/envs/project/bin/python /home/user/project/myte/src/prepare_corpus_lexicon.py "${FILE_NAME}" "${CORPUS_FILE}" --byte_patches --patch_size "${BYTE_PATCHES}"
fi



morfessor-segment "$CORPUS_FILE" -l "${OUTPUT_MODEL}" --atom-separator '-' > "${OUTPUT_FILE}"

# decode file
/home/user/miniconda3/envs/project/bin/python /home/user/project/myte/src/decode_segmented.py "${OUTPUT_FILE}" "${OUTPUT_FILE}.decoded" --escape_spaces
