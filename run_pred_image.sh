#!/usr/bin/env bash


# data processing
#ROOT_DIR="/Users/zeyesun/Documents/Data/ccks2022/task9_商品同款"
#MAIN="/Users/zeyesun/Documents/Code/torchkge/examples/train.py"
#PRETRAINED_MODEL_PATH="/Users/zeyesun/Documents/Data/bert/chinese_roberta_wwm_ext_pytorch"
ROOT_DIR="/root/autodl-tmp/Data/ccks2022/task9"
MAIN="/root/Code/commodity-alignment/pkgm/finetune_image.py"
DATA_DIR=${ROOT_DIR}/raw
OUTPUT_DIR=${ROOT_DIR}/output
MODEL_NAME="eca_nfnet_l0"
DATA_VERSION="v6.2-full"
#PRETRAINED_MODEL_PATH="/root/autodl-tmp/Data/cv/${MODEL_NAME}.pth"

THRESHOLD=0.4
EPOCH=9
EVAL_BATCH_SIZE=120

FILE_STATE_DICT="/root/autodl-tmp/Data/ccks2022/task9/output/${MODEL_NAME}-${DATA_VERSION}/image_finetune_epoch-${EPOCH}.bin"

python $MAIN \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name ${MODEL_NAME} \
  --data_version ${DATA_VERSION} \
  --config_file ${MODEL_NAME}.json \
  --file_state_dict $FILE_STATE_DICT \
  --do_pred \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --threshold $THRESHOLD \
  --image_size 800 \
  --log_steps 10 \
  --fp16
