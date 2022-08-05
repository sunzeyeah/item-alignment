#!/usr/bin/env bash


# data processing
#ROOT_DIR="/Users/zeyesun/Documents/Data/ccks2022/task9_商品同款"
#MAIN="/Users/zeyesun/Documents/Code/torchkge/examples/train.py"
#PRETRAINED_MODEL_PATH="/Users/zeyesun/Documents/Data/bert/chinese_roberta_wwm_ext_pytorch"
ROOT_DIR="/root/autodl-tmp/Data/ccks2022/task9"
MAIN="/root/Code/commodity-alignment/pkgm/finetune_text.py"
DATA_DIR=${ROOT_DIR}
OUTPUT_DIR=${ROOT_DIR}/output
MODEL_NAME="roberta_large"
DATA_VERSION="v3.4-full"
PRETRAINED_MODEL_PATH="/root/autodl-tmp/Data/bert/${MODEL_NAME}"

INTERACTION_TYPE="one_tower"
CLASSIFICATION_METHOD="cls"
SIMILARITY_MEASURE="NA"
LOSS_TYPE="ce"
THRESHOLD=0.4
EPOCH=9
EVAL_BATCH_SIZE=256

FILE_STATE_DICT="/root/autodl-tmp/Data/ccks2022/task9/output/${MODEL_NAME}-${DATA_VERSION}-${INTERACTION_TYPE}-${CLASSIFICATION_METHOD}-${SIMILARITY_MEASURE}-${LOSS_TYPE}/pkgm_finetune_epoch-${EPOCH}.bin"

python $MAIN \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name ${MODEL_NAME} \
  --data_version ${DATA_VERSION} \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --file_state_dict $FILE_STATE_DICT \
  --config_file "${MODEL_NAME}.json" \
  --do_pred \
  --interaction_type $INTERACTION_TYPE \
  --classification_method $CLASSIFICATION_METHOD \
  --similarity_measure $SIMILARITY_MEASURE \
  --loss_type $LOSS_TYPE \
  --type_vocab_size 2 \
  --max_seq_len 50 \
  --max_seq_len_pv 205 \
  --max_pvs 30 \
  --max_position_embeddings 512 \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --threshold $THRESHOLD \
  --log_steps 10 \
  --fp16


#MAIN="/root/Code/commodity-alignment/pkgm/pred_text.py"
#python $MAIN \
#  --data_dir $DATA_DIR \
#  --output_dir $OUTPUT_DIR \
#  --model_name ${MODEL_NAME}-${DATA_VERSION} \
#  --pretrained_model_path $PRETRAINED_MODEL_PATH \
#  --config_file "${MODEL_NAME}.json" \
#  --max_seq_len 64 \
#  --eval_batch_size $EVAL_BATCH_SIZE \
#  --log_steps 10 \
#  --fp16
