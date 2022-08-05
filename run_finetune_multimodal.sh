#!/usr/bin/env bash


# data processing
#ROOT_DIR="/Users/zeyesun/Documents/Data/ccks2022/task9_商品同款"
#MAIN="/Users/zeyesun/Documents/Code/torchkge/examples/train.py"
#PRETRAINED_MODEL_PATH="/Users/zeyesun/Documents/Data/bert/chinese_roberta_wwm_ext_pytorch"
ROOT_DIR="/root/autodl-tmp/Data/ccks2022/task9"
MAIN="/root/Code/commodity-alignment/pkgm/finetune_multimodal.py"
DATA_DIR=${ROOT_DIR}
OUTPUT_DIR=${ROOT_DIR}/output
MODEL_NAME="roberta_image_large"
DATA_VERSION="v5-full"
PRETRAINED_MODEL_PATH="${OUTPUT_DIR}/${MODEL_NAME}"
#PARAMETERS_TO_FREEZE="/root/autodl-tmp/Data/ccks2022/task9/output/parameters_pkgm.json"
IMAGE_MODEL_NAME="vit_base_patch16_384"

INTERACTION_TYPE="two_tower"
CLASSIFICATION_METHOD="cls"
ENSEMBLE="begin"
LOSS_TYPE="ce"
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=64
LEARNING_RATE=5e-5
NUM_EPOCHS=10

python $MAIN \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name ${MODEL_NAME} \
  --data_version ${DATA_VERSION} \
  --config_file "${MODEL_NAME}.json" \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --do_train \
  --interaction_type $INTERACTION_TYPE \
  --classification_method $CLASSIFICATION_METHOD \
  --ensemble $ENSEMBLE \
  --loss_type $LOSS_TYPE \
  --max_seq_len 50 \
  --max_seq_len_pv 305 \
  --image_hidden_size 2304 \
  --warmup_proportion 0.3 \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $NUM_EPOCHS \
  --log_steps 10 \
  --fp16
#  --parameters_to_freeze $PARAMETERS_TO_FREEZE
