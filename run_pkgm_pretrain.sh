#!/usr/bin/env bash


# data processing
#ROOT_DIR="/Users/zeyesun/Documents/Data/ccks2022/task9_商品同款"
#MAIN="/Users/zeyesun/Documents/Code/torchkge/examples/train.py"
ROOT_DIR="/root/autodl-tmp/Data/ccks2022/task9"
MAIN="/root/Code/commodity-alignment/pkgm/pkgm_pretrain.py"
DATA_DIR=${ROOT_DIR}/processed
OUTPUT_DIR=${ROOT_DIR}/output
#MODEL_NAME="transe_epoch-{}.bin"
MODEL_NAME="pkgm_epoch-{}.bin"
TRAIN_BATCH_SIZE=32768
EVAL_BATCH_SIZE=32768
LEARNING_RATE=1e-4
NUM_EPOCHS=2000
SAVE_EPOCHS=1000
EMBEDDING_DIM=768
MARGIN=1.0

python $MAIN \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name $MODEL_NAME \
  --n_neg 3 \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $NUM_EPOCHS \
  --dim $EMBEDDING_DIM \
  --margin $MARGIN \
  --save_epochs $SAVE_EPOCHS
#  --fp16
