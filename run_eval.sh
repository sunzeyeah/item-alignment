#!/usr/bin/env bash


# data processing
#ROOT_DIR="/Users/zeyesun/Documents/Data/ccks2022/task9_商品同款"
#MAIN="/Users/zeyesun/Documents/Code/torchkge/examples/train.py"
#PRETRAINED_MODEL_PATH="/Users/zeyesun/Documents/Data/bert/chinese_roberta_wwm_ext_pytorch"
ROOT_DIR="/root/autodl-tmp/Data/ccks2022/task9"
MAIN="/root/Code/commodity-alignment/pkgm/finetune.py"
PRETRAINED_MODEL_PATH="/root/autodl-tmp/Data/bert/chinese_roberta_wwm_ext_pytorch"
FILE_STATE_DICT="/root/autodl-tmp/Data/ccks2022/task9/output/roberta_base-one_tower-vec_sim-cosine-cosine/pkgm_finetune_epoch-7.bin"
DATA_DIR=${ROOT_DIR}/processed
OUTPUT_DIR=${ROOT_DIR}/output
MODEL_NAME="roberta_base"
INTERATION_TYPE="one_tower"
CLASSIFICATION_METHOD="vec_sim"
SIMILARITY_MEASURE="cosine"
LOSS_TYPE="cosine"
MAX_SEQ_LENGTH=50
MAX_SEQ_LENGTH_PVS=205
MAX_NUM_PV=30
TYPE_VOCAB_SIZE=2
EVAL_BATCH_SIZE=108

python $MAIN \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name $MODEL_NAME \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --file_state_dict $FILE_STATE_DICT \
  --config_file "model_base.json" \
  --do_eval \
  --interaction_type $INTERATION_TYPE \
  --classification_method $CLASSIFICATION_METHOD \
  --similarity_measure $SIMILARITY_MEASURE \
  --loss_type $LOSS_TYPE \
  --type_vocab_size $TYPE_VOCAB_SIZE \
  --max_seq_len $MAX_SEQ_LENGTH \
  --max_seq_len_pv $MAX_SEQ_LENGTH_PVS \
  --max_pvs $MAX_NUM_PV \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --fp16 \
