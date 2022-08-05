#!/usr/bin/env bash


# data processing
ROOT_DIR="/root/autodl-tmp/Data/ccks2022/task9"
MAIN="/root/Code/commodity-alignment/pkgm/model_soup_text.py"
DATA_DIR=${ROOT_DIR}/processed
OUTPUT_DIR=${ROOT_DIR}/output
MODEL_NAME="roberta_large"
DATA_VERSION="v3.4-full"
PRETRAINED_MODEL_PATH="/root/autodl-tmp/Data/bert/${MODEL_NAME}"

INTERACTION_TYPE="one_tower"
CLASSIFICATION_METHOD="cls"
SIMILARITY_MEASURE="NA"
LOSS_TYPE="ce"
BATCH_SIZE=40

FILE_STATE_DICT="/root/autodl-tmp/Data/ccks2022/task9/output/${MODEL_NAME}-${DATA_VERSION}-${INTERACTION_TYPE}-${CLASSIFICATION_METHOD}-${SIMILARITY_MEASURE}-${LOSS_TYPE}/pkgm_finetune_epoch-{}.bin"
EPOCHS="6,7,8,9"


python $MAIN \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name ${MODEL_NAME} \
  --data_version ${DATA_VERSION} \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --config_file "${MODEL_NAME}.json" \
  --interaction_type $INTERACTION_TYPE \
  --classification_method $CLASSIFICATION_METHOD \
  --similarity_measure $SIMILARITY_MEASURE \
  --loss_type $LOSS_TYPE \
  --file_state_dict $FILE_STATE_DICT \
  --epochs $EPOCHS \
  --max_seq_len 50 \
  --max_seq_len_pv 205 \
  --max_pvs 30 \
  --eval_batch_size $BATCH_SIZE \
  --threshold 0.5 \
  --log_steps 10 \
  --fp16 \
