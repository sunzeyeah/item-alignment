#!/usr/bin/env bash


ROOT_DIR="${HOME}/Data"
DATA_DIR=${ROOT_DIR}
OUTPUT_DIR=${ROOT_DIR}/output
PRETRAINED_MODEL_PATH="${ROOT_DIR}/bert/roberta_large"


# Roberta_large-v3.4
FILE_STATE_DICT="${HOME}/Data/output/roberta_large-v3.4-one_tower-cls-NA-ce/text_finetune_epoch-9.bin"
python finetune_text.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name "roberta_large" \
  --data_version "v3.4" \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --file_state_dict $FILE_STATE_DICT \
  --config_file "src/config/roberta_large.json" \
  --do_pred \
  --interaction_type "one_tower" \
  --classification_method "cls" \
  --similarity_measure "NA" \
  --loss_type "ce" \
  --max_seq_len 50 \
  --max_seq_len_pv 205 \
  --eval_batch_size 108 \
  --threshold 0.4 \
  --fp16

# Roberta_large-v3.4-cls_cat_1,2,3,4
FILE_STATE_DICT="${HOME}/Data/output/roberta_large-v3.4-one_tower-cls_1,2,3,4_cat-NA-ce/text_finetune_epoch-9.bin"
python finetune_text.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name "roberta_large" \
  --data_version "v3.4" \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --file_state_dict $FILE_STATE_DICT \
  --config_file "src/config/roberta_large.json" \
  --do_pred \
  --interaction_type "one_tower" \
  --classification_method "cls_1,2,3,4_cat" \
  --similarity_measure "NA" \
  --loss_type "ce" \
  --cls_layers "1,2,3,4" \
  --cls_pool "cat" \
  --max_seq_len 50 \
  --max_seq_len_pv 205 \
  --eval_batch_size 108 \
  --threshold 0.4 \
  --fp16

# Roberta_large-v4
FILE_STATE_DICT="${HOME}/Data/output/roberta_large-v4-one_tower-cls-NA-ce/text_finetune_epoch-9.bin"
python finetune_text.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name "roberta_large" \
  --data_version "v4" \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --file_state_dict $FILE_STATE_DICT \
  --config_file "src/config/roberta_large.json" \
  --do_pred \
  --interaction_type "one_tower" \
  --classification_method "cls" \
  --similarity_measure "NA" \
  --loss_type "ce" \
  --max_seq_len 50 \
  --max_seq_len_pv 205 \
  --eval_batch_size 108 \
  --threshold 0.4 \
  --fp16

# pkgm_large-v3.4
FILE_STATE_DICT="${HOME}/Data/output/pkgm_large-v3.4-one_tower-cls-NA-ce/text_finetune_epoch-9.bin"
python finetune_text.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name "pkgm_large" \
  --data_version "v3.4" \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --file_state_dict $FILE_STATE_DICT \
  --config_file "src/config/pkgm_large.json" \
  --do_pred \
  --interaction_type "one_tower" \
  --classification_method "cls" \
  --similarity_measure "NA" \
  --loss_type "ce" \
  --max_seq_len 64 \
  --max_pvs 30 \
  --eval_batch_size 512 \
  --threshold 0.4 \
  --fp16

# textcnn-v3.4
FILE_STATE_DICT="${HOME}/Data/output/textcnn-v3.4-two_tower-cls-NA-ce/text_finetune_epoch-9.bin"
python finetune_text.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name "textcnn" \
  --data_version "v3.4" \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --file_state_dict $FILE_STATE_DICT \
  --config_file "src/config/roberta_large.json" \
  --do_pred \
  --interaction_type "two_tower" \
  --classification_method "cls" \
  --similarity_measure "NA" \
  --loss_type "ce" \
  --max_seq_len 50 \
  --max_seq_len_pv 205 \
  --eval_batch_size 512 \
  --threshold 0.4 \
  --fp16

# bert_base
python pred_bert.py

# roberta_image_large-v5
FILE_STATE_DICT="${HOME}/Data/output/roberta_image_large-v5-one_tower-cls-begin-ce/multimodal_finetune_epoch-9.bin"
python finetune_multimodal.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name "roberta_image_large" \
  --data_version "v5" \
  --config_file "src/config/roberta_image_large.json" \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --file_state_dict $FILE_STATE_DICT \
  --do_pred \
  --interaction_type "one_tower" \
  --classification_method "cls" \
  --ensemble "begin" \
  --loss_type "ce" \
  --max_seq_len 50 \
  --max_seq_len_pv 205 \
  --eval_batch_size 108 \
  --threshold 0.4 \
  --fp16

# eca_nfnet_l0-v6
FILE_STATE_DICT="${HOME}/Data/output/eca_nfnet_l0-v6/image_finetune_epoch-9.bin"
python finetune_image.py \
  --data_dir $DATA_DIR/raw \
  --output_dir $OUTPUT_DIR \
  --model_name "eca_nfnet_l0" \
  --data_version "v6" \
  --config_file "src/config/eca_nfnet_l0.json" \
  --file_state_dict $FILE_STATE_DICT \
  --do_pred \
  --image_size 1000 \
  --eval_batch_size 128 \
  --threshold 0.4 \
  --fp16

# model ensemble
python model_ensemble.py \
  --data_dir $DATA_DIR \
  --ensemble_strategy "threshold" \
  --split_by_valid_or_test