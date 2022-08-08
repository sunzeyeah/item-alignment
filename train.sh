#!/usr/bin/env bash


ROOT_DIR="${HOME}/Data"
DATA_DIR=${ROOT_DIR}
OUTPUT_DIR="${ROOT_DIR}/output"
PRETRAINED_MODEL_PATH="${ROOT_DIR}/bert"

# Roberta_large-v3.4
python finetune_text.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name "roberta_large" \
  --data_version "v3.4-full" \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --config_file "config/roberta_large.json" \
  --do_train \
  --interaction_type "one_tower" \
  --classification_method "cls" \
  --similarity_measure "NA" \
  --loss_type "ce" \
  --max_seq_len 50 \
  --max_seq_len_pv 205 \
  --train_batch_size 40 \
  --learning_rate 5e-5 \
  --fp16

# Roberta_large-v3.4-cls_cat_1,2,3,4
python finetune_text.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name "roberta_large" \
  --data_version "v3.4-full" \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --config_file "config/roberta_large.json" \
  --do_train \
  --interaction_type "one_tower" \
  --classification_method "cls_1,2,3,4_cat" \
  --similarity_measure "NA" \
  --loss_type "ce" \
  --cls_layers "1,2,3,4" \
  --cls_pool "cat" \
  --max_seq_len 50 \
  --max_seq_len_pv 205 \
  --train_batch_size 40 \
  --learning_rate 5e-5 \
  --fp16

# Roberta_large-v4
python finetune_text.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name "roberta_large" \
  --data_version "v4-full" \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --config_file "config/roberta_large.json" \
  --do_train \
  --interaction_type "one_tower" \
  --classification_method "cls" \
  --similarity_measure "NA" \
  --loss_type "ce" \
  --max_seq_len 50 \
  --max_seq_len_pv 205 \
  --train_batch_size 40 \
  --learning_rate 5e-5 \
  --fp16

# pkgm_large-v3.4
python finetune_text.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name "pkgm_large" \
  --data_version "v3.4-full" \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --config_file "config/pkgm_large.json" \
  --do_train \
  --interaction_type "one_tower" \
  --classification_method "cls" \
  --similarity_measure "NA" \
  --loss_type "ce" \
  --max_seq_len 64 \
  --max_pvs 30 \
  --train_batch_size 256 \
  --learning_rate 5e-5 \
  --fp16

# textcnn-v3.4
python finetune_text.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name "textcnn" \
  --data_version "v3.4-full" \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --config_file "config/roberta_large.json" \
  --do_train \
  --interaction_type "two_tower" \
  --classification_method "cls" \
  --similarity_measure "NA" \
  --loss_type "ce" \
  --max_seq_len 50 \
  --max_seq_len_pv 205 \
  --train_batch_size 256 \
  --learning_rate 5e-5 \
  --fp16

# bert_base
python src/bert/run_train.py

# roberta_image_large-v5
python finetune_multimodal.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name "roberta_image_large" \
  --data_version "v5-full" \
  --config_file "config/roberta_image_large.json" \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --do_train \
  --interaction_type "one_tower" \
  --classification_method "cls" \
  --ensemble "begin" \
  --loss_type "ce" \
  --max_seq_len 50 \
  --max_seq_len_pv 205 \
  --train_batch_size 40 \
  --learning_rate 5e-5 \
  --fp16

# eca_nfnet_l0-v6
python finetune_image.py \
  --data_dir $DATA_DIR/raw \
  --output_dir $OUTPUT_DIR \
  --model_name "eca_nfnet_l0" \
  --data_version "v6-full" \
  --config_file "config/eca_nfnet_l0.json" \
  --do_train \
  --image_size 1000 \
  --train_batch_size 64 \
  --learning_rate 5e-5 \
  --fp16

