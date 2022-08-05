#!/usr/bin/env bash


# data processing
#ROOT_DIR="/Users/zeyesun/Documents/Data/ccks2022/task9_商品同款"
#MAIN="/Users/zeyesun/Documents/Code/torchkge/examples/train.py"
ROOT_DIR="/root/autodl-tmp/Data/ccks2022/task9"
MAIN="/root/Code/commodity-alignment/pkgm/data_prepare.py"
DATA_DIR=${ROOT_DIR}/raw
OUTPUT_DIR=${ROOT_DIR}/processed
NUM_TRAIN_AUGMENT=0
#NUM_TRAIN_AUGMENT=22000
#NUM_TRAIN_AUGMENT=49000

# ONLY IMAGE DATA
#python $MAIN \
#  --data_dir $DATA_DIR \
#  --output_dir $OUTPUT_DIR \
#  --dtypes "train,valid" \
#  --only_image \
#  --image_size 1000

# TEXT DATA
#python $MAIN \
#  --data_dir $DATA_DIR \
#  --output_dir $OUTPUT_DIR \
#  --dtypes "train,valid" \
#  --filter_method "freq" \
#  --min_freq 10 \
#  --min_prop 0.5 \
#  --num_train_augment $NUM_TRAIN_AUGMENT \
#  --num_neg 1 \
#  --split_on_train \
#  --prev_valid $OUTPUT_DIR/finetune_train_valid_orig.tsv \
#  --valid_proportion 0.25 \
#  --valid_pos_proportion 0.4
##  --with_image \
##  --cv_model_name "eca_nfnet_l0" \
##  --finetuned \
##  --pretrained_model_path "/root/autodl-tmp/Data/ccks2022/task9/output/eca_nfnet_l0-v6-full/image_finetune_epoch-4.bin" \
##  --image_size 1000 \
##  --batch_size 256

# IMAGE OBJECT DETECTION
python $MAIN \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --dtypes "train,valid" \
  --only_image \
  --object_detection \
  --cv_model_name "yolov5x6" \
  --code_path "/root/Code/yolov5" \
  --pretrained_model_path "/root/autodl-tmp/Data/cv/yolov5x6.pt" \
  --min_crop_ratio 0.1
