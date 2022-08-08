# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :EntityAlignNet
# @File     :data_bert_utils
# @Date     :2022/6/29 11:40
# @Author   :mengqingyang
# @Email    :mengqingyang0102@163.com
-------------------------------------------------
"""

import os
import json

import torch
import numpy as np
from random import shuffle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from bert.log import LOGGER


def read_data(data_dir, file, pair_names):
    records = []
    file_name = os.path.join(data_dir, file)
    with open(file_name, 'r', encoding="utf8") as f:
        data = json.loads(f.read())
        if "data" in data:
            for rec in data["data"]:
                records.append([rec.get(pair_names[0], ""), rec.get(pair_names[1], ""), int(rec["label"])])
        else:
            for rec in data:
                records.append([rec.get(pair_names[0], ""), rec.get(pair_names[1], ""), int(rec["label"])])
    return records


def shuffle_pvs_pairs(pvs_pairs):
    shuffled_pvs = []
    for pvs_a, pvs_b, label in pvs_pairs:
        pvs_a_ = pvs_a.split(";")
        shuffle(pvs_a_)

        pvs_b_ = pvs_b.split(";")
        shuffle(pvs_b_)

        rand = np.random.random()
        if rand < 0.5:
            shuffled_pvs.append([';'.join(pvs_b_), ";".join(pvs_a_), label])
        else:
            shuffled_pvs.append([';'.join(pvs_a_), ";".join(pvs_b_), label])
    return shuffled_pvs


def join_data(data_dir, filename, do_shuffle=True):
    pvs_pairs = read_data(data_dir, filename, ["pvs_a", "pvs_b"])
    if do_shuffle:
        pvs_pairs = shuffle_pvs_pairs(pvs_pairs)
    title_pairs = read_data(data_dir, filename, ["title_a", "title_b"])
    industry_name_pairs = read_data(data_dir, filename, ["industry_name_a", "industry_name_b"])
    cate_pairs = read_data(data_dir, filename, ["cate_a", "cate_b"])
    cate_path_pairs = read_data(data_dir, filename, ["cate_path_a", "cate_path_b"])
    return pvs_pairs, title_pairs, industry_name_pairs, cate_pairs, cate_path_pairs


def show(data, mode, name, show_num=3):
    LOGGER.info("")
    LOGGER.info(f"======= {mode} / {name} ==========")
    for pv in data[:show_num]:
        LOGGER.info(pv)
    LOGGER.info(f"======= *******^_^******* ==========")


def get_examples(pvs, titles, cates, cate_paths, industry_names):
    pvs_src, pvs_tgt, labels = zip(*pvs)
    titles_src, titles_tgt, labels = zip(*titles)
    cate_src, cate_tgt, labels = zip(*cates)
    cate_path_src, cate_path_tgt, labels = zip(*cate_paths)
    industry_name_src, industry_name_tgt, labels = zip(*industry_names)
    return pvs_src, pvs_tgt, titles_src, titles_tgt, cate_src, cate_tgt, cate_path_src, cate_path_tgt, industry_name_src, industry_name_tgt, labels


def show_pairs(src_data, tgt_data, labels, name, mode="train", show_num=3):
    LOGGER.info("")
    LOGGER.info(f"======= {mode} / {name} ==========")
    for a, b, label in zip(src_data[:show_num], tgt_data[:show_num], labels[:show_num]):
        LOGGER.info(f"src_{name}:" + str(a))
        LOGGER.info(f"tgt_{name}:" + str(b))
        LOGGER.info("label:" + str(label))
    LOGGER.info(f"======= *******^_^*******  ==========")


def encode(tokenizer, pvs_src, pvs_tgt, title_src, title_tgt, cate_src, cate_tgt, cate_path_src, cate_path_tgt,
                industry_name_src,
                industry_name_tgt, pvs_len=512, title_len=150, cate_len=20, cate_path_len=50, industry_name_len=20):
    pvs = tokenizer(pvs_src, pvs_tgt,
                    padding='max_length', truncation=True, max_length=pvs_len)

    LOGGER.info("pvs encoded ^_^")
    title = tokenizer(title_src, title_tgt,
                      padding='max_length', truncation=True, max_length=title_len)
    LOGGER.info("title encoded ^_^")
    cate = tokenizer(cate_src, cate_tgt,
                     padding='max_length', truncation=True, max_length=cate_len)
    LOGGER.info("cate encoded ^_^")
    cate_path = tokenizer(cate_path_src, cate_path_tgt,
                          padding='max_length', truncation=True, max_length=cate_path_len)
    LOGGER.info("cate_path encoded ^_^")
    industry_name = tokenizer(industry_name_src, industry_name_tgt, padding='max_length',
                              truncation=True, max_length=industry_name_len)
    LOGGER.info("industry_name encoded ^_^")
    return pvs, title, cate, cate_path, industry_name


def convert_examples_to_features(example):
    input_ids = torch.tensor(example['input_ids'])
    attention_mask = torch.tensor(example['attention_mask'])
    token_type_ids = torch.tensor(example['token_type_ids'])
    return input_ids, attention_mask, token_type_ids


def get_dataloader(pvs, titles, cates, cate_paths, industry_names, labels, batch_size=4, mode="train"):
    pvs_input_ids, pvs_attention_mask, pvs_token_type_ids = convert_examples_to_features(pvs)
    title_input_ids, title_attention_mask, title_token_type_ids = convert_examples_to_features(titles)
    cate_input_ids, cate_attention_mask, cate_token_type_ids = convert_examples_to_features(cates)
    cate_path_input_ids, cate_path_attention_mask, cate_path_token_type_ids = convert_examples_to_features(cate_paths)
    industry_name_input_ids, industry_name_attention_mask, industry_name_token_type_ids = convert_examples_to_features(industry_names)
    tensor_labels = torch.tensor(labels)

    # Create the DataLoader.
    data = TensorDataset(pvs_input_ids, pvs_attention_mask, pvs_token_type_ids,
                         title_input_ids, title_attention_mask, title_token_type_ids,
                         cate_input_ids, cate_attention_mask, cate_token_type_ids,
                         cate_path_input_ids, cate_path_attention_mask, cate_path_token_type_ids,
                         industry_name_input_ids, industry_name_attention_mask, industry_name_token_type_ids,
                         tensor_labels)
    if mode == "train":
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

