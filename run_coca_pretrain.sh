#!/usr/bin/env bash

# data processing
ROOT_DIR="/root/autodl-tmp/Data/ccks2022/task9"
MAIN="/root/Code/commodity-alignment/pkgm/coca_pretrain.py"
DATA_DIR=${ROOT_DIR}/raw
OUTPUT_DIR=${ROOT_DIR}/output
MODEL_NAME="coca_base"
PRETRAINED_TEXT_MODEL_PATH="/root/autodl-tmp/Data/bert/roberta_base"
PRETRAINED_IMAGE_MODEL_PATH="/root/autodl-tmp/Data/cv/vit_base_patch16_384/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz"
#PRETRAINED_IMAGE_MODEL_PATH="/root/autodl-tmp/Data/cv/vit_large_patch16_384/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz"

IMAGE_MODEL_NAME="vit_base_patch16_384"
TRAIN_BATCH_SIZE=108
LEARNING_RATE=1e-4
NUM_EPOCHS=10

python $MAIN \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name $MODEL_NAME \
  --config_file ${MODEL_NAME}.json \
  --pretrained_text_model_path $PRETRAINED_TEXT_MODEL_PATH \
  --pretrained_image_model_path $PRETRAINED_IMAGE_MODEL_PATH \
  --image_model_name $IMAGE_MODEL_NAME \
  --image_size 384 \
  --max_seq_len 64 \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --warmup_proportion 0.3 \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $NUM_EPOCHS \
  --log_steps 10 \
  --fp16
