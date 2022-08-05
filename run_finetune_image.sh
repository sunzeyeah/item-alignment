#!/usr/bin/env bash


# data processing
ROOT_DIR="/root/autodl-tmp/Data/ccks2022/task9"
MAIN="/root/Code/commodity-alignment/pkgm/finetune_image.py"
DATA_DIR=${ROOT_DIR}/raw
OUTPUT_DIR=${ROOT_DIR}/output
MODEL_NAME="vit_base_patch16_384"
DATA_VERSION="v6"
#PRETRAINED_MODEL_PATH="/root/autodl-tmp/Data/cv/${MODEL_NAME}.pth"
PRETRAINED_MODEL_PATH="/root/autodl-tmp/Data/cv/vit_base_patch16_384/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz"
#PRETRAINED_MODEL_PATH="/root/autodl-tmp/Data/cv/vit_large_patch16_384/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz"

TRAIN_BATCH_SIZE=56
EVAL_BATCH_SIZE=112
LEARNING_RATE=5e-5
NUM_EPOCHS=10

python $MAIN \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name ${MODEL_NAME} \
  --data_version ${DATA_VERSION} \
  --config_file ${MODEL_NAME}.json \
  --do_train \
  --do_eval \
  --warmup_proportion 0.3 \
  --image_size 384 \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $NUM_EPOCHS \
  --log_steps 10 \
  --fp16
#  --pretrained_model_path $PRETRAINED_MODEL_PATH \
