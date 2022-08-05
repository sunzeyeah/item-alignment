#!/usr/bin/env bash


# data processing
ROOT_DIR="/root/autodl-tmp/Data/ccks2022/task9"
MAIN="/root/Code/commodity-alignment/pkgm/finetune_graph.py"
DATA_DIR=${ROOT_DIR}
OUTPUT_DIR=${ROOT_DIR}/output
MODEL_NAME="gcn"
DATA_VERSION="v1"
PRETRAINED_MODEL_PATH="/root/autodl-tmp/Data/bert/roberta_large"
#PARAMETERS_TO_FREEZE="/root/autodl-tmp/Data/ccks2022/task9/output/parameters_pkgm.json"
#PARAMETERS_TO_FREEZE="/root/autodl-tmp/Data/ccks2022/task9/output/textcnn_parameters_to_freeze.json"

INTERACTION_TYPE="two_tower"
CLASSIFICATION_METHOD="cls"
SIMILARITY_MEASURE="NA"
LOSS_TYPE="ce"
BATCH_SIZE=512
LEARNING_RATE=1e-4
NUM_EPOCHS=500

python $MAIN \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name ${MODEL_NAME} \
  --data_version ${DATA_VERSION} \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --config_file "${MODEL_NAME}.json" \
  --do_train \
  --do_eval \
  --interaction_type $INTERACTION_TYPE \
  --classification_method $CLASSIFICATION_METHOD \
  --similarity_measure $SIMILARITY_MEASURE \
  --loss_type $LOSS_TYPE \
  --warmup_proportion 0.3 \
  --train_batch_size $BATCH_SIZE \
  --eval_batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $NUM_EPOCHS \
  --num_layers 4 \
  --hidden_size 128 \
  --log_steps 10 \
  --save_epochs 10
#  --fp16
#  --auxiliary_task
#  --parameters_to_freeze $PARAMETERS_TO_FREEZE
