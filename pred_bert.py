# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :EntityAlignNet
# @File     :bert_encoder
# @Date     :2022/7/1 09:20
# @Author   :mengqingyang
# @Email    :mengqingyang0102@163.com
-------------------------------------------------
"""

# -*- coding: utf-8 -*-
import os
# import sys
# sys.path.append("..")
import math
import json
from typing import List
import numpy as np
import time
import datetime
import torch
import random
import transformers
from transformers import BertTokenizer

from src.bert.log import LOGGER
from src.bert.data_utils import read_data, join_data, get_examples, show_pairs, encode, get_dataloader
# from .finetune_bert import evaluate
from src.bert.model import BertAlignModel

HOME = os.path.expanduser("~")
# HOME = os.getenv("HOME")
DATA_DIR = os.path.join(HOME, "Data", "processed")
OUTPUT_DIR = os.path.join(HOME, "Data", "output", "bert_base-one_tower-cls-NA-ce")
MODEL_DIR = os.path.join(OUTPUT_DIR, "F1Model", "pytorch_model.bin")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def compute(item_emb_1: List[float], item_emb_2: List[float], bias=0.0) -> float:
    s = 0.0
    for a, b in zip(item_emb_1, item_emb_2):
        s += a * b
    s += bias
    return s


def metrics(preds, labels):
    tp, tn, fp, fn = 0, 0, 0, 0
    for p, t in zip(preds, labels):
        if p == t and t == 1:
            tp += 1
        if p == t and t == 0:
            tn += 1
        if p != t and t == 1:
            fn += 1
        if p != t and t == 0:
            fp += 1
    acc = (tp + tn) / (tp + tn + fp + fn + 0.0001)
    pre = tp / (tp + fp + 0.0001)
    recall = tp / (tp + fn + 0.0001)
    F1 = (2 * pre * recall) / (recall + pre + 0.0001)
    print("acc:", acc, "pre:", pre, "recall:", recall, "F1:", F1)
    return acc, pre, recall, F1


def format_time(elapsed):
    '''Takes a time in seconds and returns a string hh:mm:ss'''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def load_model(model, model_dir):
    need_model_dict = model.state_dict()
    have_model_state = torch.load(model_dir, map_location="cpu")
    new_dict = {}
    for attr in have_model_state:
        if attr.startswith("module."):
            attr = attr.replace("module.", "", 1)  # 先改名
            if attr in need_model_dict:  # 需要
                new_dict[attr] = have_model_state["module." + attr]
        else:
            if attr in need_model_dict:  # 需要
                new_dict[attr] = have_model_state[attr]

    need_model_dict.update(new_dict)  # 更新对应的值
    model.load_state_dict(need_model_dict)
    del have_model_state  # 这里，手动释放cpu内存...
    del new_dict
    LOGGER.info(f'Load existed model from {model_dir} successfully ^_^')
    return model


def main():
    # 验证cuda
    # model_dir = "/root/autodl-tmp/EntityAlignNet/finetunes/bert-old/F1Model/pytorch_model.bin"
    # data_dir = "/root/autodl-tmp/corp/"
    val_file = "item-align-val.json"
    test_file = "item-align-test.json"
    # encode_output_file = "./deepAI_v1.jsonl"
    encode_output_file = os.path.join(OUTPUT_DIR, "deepAI_result_threshold=0.4.jsonl")
    batch_size = 16
    test_data_ids = read_data(DATA_DIR, test_file, ["text_id_a", "text_id_b"])
    # pvs_train, title_train, industry_name_train, cate_train, cate_path_train = join_data(data_dir, train_file)
    pvs_eval, title_eval, industry_name_eval, cate_eval, cate_path_eval = join_data(DATA_DIR, val_file, do_shuffle=False)
    pvs_test, title_test, industry_name_test, cate_test, cate_path_test = join_data(DATA_DIR, test_file, do_shuffle=False)

    pvs_src_eval, pvs_tgt_eval, titles_src_eval, titles_tgt_eval, cate_src_eval, cate_tgt_eval, cate_path_src_eval, cate_path_tgt_eval, \
    industry_name_src_eval, industry_name_tgt_eval, eval_labels = get_examples(pvs_eval,
                                                                               title_eval,
                                                                               cate_eval,
                                                                               cate_path_eval,
                                                                               industry_name_eval)
    show_pairs(pvs_src_eval, pvs_tgt_eval, eval_labels, "pvs", mode="eval")
    show_pairs(titles_src_eval, titles_tgt_eval, eval_labels, "title", mode="eval")
    show_pairs(cate_src_eval, cate_tgt_eval, eval_labels, "cate", mode="eval")
    show_pairs(cate_path_src_eval, cate_path_tgt_eval, eval_labels, "cate_path", mode="eval")
    show_pairs(industry_name_src_eval, industry_name_tgt_eval, eval_labels, "industry_name", mode="eval")

    pvs_src_test, pvs_tgt_test, titles_src_test, titles_tgt_test, cate_src_test, cate_tgt_test, cate_path_src_test, cate_path_tgt_test, \
    industry_name_src_test, industry_name_tgt_test, test_labels = get_examples(pvs_test,
                                                                               title_test,
                                                                               cate_test,
                                                                               cate_path_test,
                                                                               industry_name_test)
    show_pairs(pvs_src_test, pvs_tgt_test, test_labels, "pvs", mode="test")
    show_pairs(titles_src_test, titles_tgt_test, test_labels, "title", mode="test")
    show_pairs(cate_src_test, cate_tgt_test, test_labels, "cate", mode="test")
    show_pairs(cate_path_src_test, cate_path_tgt_test, test_labels, "cate_path", mode="test")
    show_pairs(industry_name_src_test, industry_name_tgt_test, test_labels, "industry_name", mode="test")

    transformers.logging.set_verbosity_error()
    # Load the BERT tokenizer.
    LOGGER.info('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
#     tokenizer.do_basic_tokenize = False

    LOGGER.info("======== Encode Eval Data ==========")
    pvs_eval, title_eval, cate_eval, cate_path_eval, industry_name_eval = encode(tokenizer,
                                                                                      pvs_src_eval, pvs_tgt_eval,
                                                                                      titles_src_eval, titles_tgt_eval,
                                                                                      cate_src_eval, cate_tgt_eval,
                                                                                      cate_path_src_eval,
                                                                                      cate_path_tgt_eval,
                                                                                      industry_name_src_eval,
                                                                                      industry_name_tgt_eval)
    LOGGER.info("======== Encode Test Data ==========")
    pvs_test, title_test, cate_test, cate_path_test, industry_name_test = encode(tokenizer,
                                                                                      pvs_src_test, pvs_tgt_test,
                                                                                      titles_src_test, titles_tgt_test,
                                                                                      cate_src_test, cate_tgt_test,
                                                                                      cate_path_src_test,
                                                                                      cate_path_tgt_test,
                                                                                      industry_name_src_test,
                                                                                      industry_name_tgt_test)

    eval_dataloader = get_dataloader(pvs_eval, title_eval, cate_eval, cate_path_eval, industry_name_eval, eval_labels, batch_size=batch_size)
    test_dataloader = get_dataloader(pvs_test, title_test, cate_test, cate_path_test, industry_name_test, test_labels, batch_size=batch_size, mode="test")

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    model = BertAlignModel.from_pretrained("bert-base-chinese")
    model = load_model(model, MODEL_DIR)
    model.to(device)
    # model.cuda()
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    LOGGER.info('The BERT model has {:} different named parameters.\n'.format(len(params)))
    LOGGER.info('==== Embedding Layer ====\n')
    for p in params[0:5]:
        LOGGER.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    LOGGER.info('\n==== First Transformer ====\n')
    for p in params[5:21]:
        LOGGER.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    LOGGER.info('\n==== Output Layer ====\n')
    for p in params[-4:]:
        LOGGER.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    # LOGGER.info("====== Evaluate the current model ======")
    # F1, pre, recall, acc, cls_F1, cls_pre, cls_recall, cls_acc, eval_loss = evaluate(model, eval_dataloader, device, -1, -1)

    # acc, pre, recall, F1 = evaluate(model, eval_dataloader, device)
    # LOGGER.info(
    #     "* Accuracy: {0:>5.4}, Precision: {1:>5.4}, Recall: {2:>5.4}, F1: {3:.>5.4} ^_^".format(
    #         acc, pre, recall, F1))

    # ========================================
    #               Encode
    # ========================================
    torch.cuda.empty_cache()
    LOGGER.info('======== Predict Test Data ========')
    t0 = time.time()
    encode_embeddings = []
    for step, batch in enumerate(test_dataloader):
        d = tuple(t.to(device) for t in batch)
        pvs_input_ids, pvs_input_mask, pvs_token_ids, title_input_ids, title_input_mask, title_token_ids, cate_input_ids, cate_input_mask, cate_token_ids, cate_path_input_ids, cate_path_input_mask, cate_path_token_ids, industry_name_input_ids, industry_name_input_mask, industry_name_token_ids, labels = d
        output = model(pvs_input_ids=pvs_input_ids, pvs_token_type_ids=pvs_token_ids, pvs_attention_mask=pvs_input_mask,
                       title_input_ids=title_input_ids, title_token_type_ids=title_token_ids, title_attention_mask=title_input_mask,
                       cate_input_ids=cate_input_ids, cate_token_type_ids=cate_token_ids, cate_attention_mask=cate_input_mask,
                       cate_path_input_ids=cate_path_input_ids, cate_path_token_type_ids=cate_path_token_ids, cate_path_attention_mask=cate_path_input_mask,
                       industry_name_input_ids=industry_name_input_ids, industry_name_token_type_ids=industry_name_token_ids, industry_name_attention_mask=industry_name_input_mask)
        pool_out = output[0].detach().cpu()
        encode_embeddings.append(pool_out)

        if step % 1000 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            LOGGER.info(f"predict {step} ...")
    encode_embeddings = torch.cat(encode_embeddings, dim=0)
    assert len(test_data_ids) == encode_embeddings.shape[0]
    weight, bias = model.get_sim_eval_weight()
    preds = []
    labels = []
    with open(encode_output_file, "w", encoding="utf-8") as writer:
        for record, embedding in zip(test_data_ids, encode_embeddings):
            score = (weight * embedding).sum() + bias
            # rec["src_item_emb"] = ",".join([str(d.item()) for d in list(embedding)])
            # rec["tgt_item_emb"] = ",".join([str(d.item()) for d in list(weight)])
            threshold = -bias.item() # float(record['threshold'])
            prob = 1 / (1 + math.exp(-score + threshold))
            rec = {
                "src_item_id": record[0],
                "tgt_item_id": record[1],
                "src_item_emb": str([1 - prob]),
                "tgt_item_emb": str([prob]),
                "threshold": 0.4
            }
            writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if score.item() > 0.0:
                preds.append(1.0)
            else:
                preds.append(0.0)
            labels.append(record[-1])
    metrics(preds, labels)


if __name__ == "__main__":
    main()




