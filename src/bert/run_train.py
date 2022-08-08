# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :EntityAlignNet
# @File     :run_bert_train
# @Date     :2022/6/29 11:30
# @Author   :mengqingyang
# @Email    :mengqingyang0102@163.com
-------------------------------------------------
"""
# import sys
# sys.path.append("..")
import numpy as np
import time
import datetime
import torch
import random
from tensorboardX import SummaryWriter
import os
import csv
import logging
import transformers
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup, BertTokenizer

from .data_utils import join_data, get_examples, get_dataloader, show, show_pairs, encode
from .model import BertAlignModel
from .log import LOGGER

HOME = os.path.expanduser("~")
# HOME = os.getenv("HOME")
DATA_DIR = os.path.join(HOME, "Data", "processed")
OUTPUT_DIR = os.path.join(HOME, "Data", "output", "bert_base-one_tower-cls-NA-ce")
LOG_DIR = os.path.join(OUTPUT_DIR, "tf-logs")


class EvalWriter(object):
    def __init__(self, path="./"):
        self.output_path = path
        self.csv_file = "bert_align_results.csv"
        self.csv_headers = ["accuracy", "f1", "precision", "recall", "threshold",
                            "classify_accuracy", "classify_f1", "classify_precision", "classify_recall",
                            "classify_threshold",
                            "epoch", "steps"]

    def update(self,
               f1, precision, recall, acc, threshold,
               classify_f1, classify_precision, classify_recall, classify_acc, classify_threshold,
               epoch, steps):
        csv_path = os.path.join(self.output_path, self.csv_file)
        if not os.path.isfile(csv_path):
            with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)
                writer.writerow([acc, f1, precision, recall, threshold,
                                 classify_f1, classify_precision, classify_recall, classify_acc, classify_threshold,
                                 epoch, steps])
        else:
            with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([acc, f1, precision, recall, threshold,
                                 classify_acc, classify_f1, classify_precision, classify_recall, classify_threshold,
                                 epoch, steps])


EVAL_WRITER = EvalWriter()


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool = True):
    assert len(scores) == len(labels)
    rows = list(zip(scores, labels))

    rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

    best_f1 = best_precision = best_recall = best_acc = 0
    threshold = 0
    nextract = 0
    ncorrect = 0
    fneg = 0
    total_num_duplicates = sum(labels)
    neg_total = len(labels) - total_num_duplicates
    for i in range(len(rows) - 1):
        score, label = rows[i]
        nextract += 1

        if label == 1:
            ncorrect += 1
        else:
            fneg += 1

        if ncorrect > 0:
            precision = ncorrect / nextract
            recall = ncorrect / total_num_duplicates
            f1 = 2 * precision * recall / (precision + recall)
            acc = (ncorrect + neg_total - fneg) / len(labels)
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                threshold = (rows[i][0] + rows[i + 1][0]) / 2
                best_acc = acc

    return best_acc, best_f1, best_precision, best_recall, threshold


def flat_preds_and_labels(preds, labels):
    pred_labels, real_labels = [], []
    pred_flat = preds
    pred_argmax_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    tp = tn = fp = fn = 0
    for pf, p, t in zip(pred_flat, pred_argmax_flat, labels_flat):
        pred_labels.append(pf[1] - pf[0])
        real_labels.append(t)
        if p == t and t == 1:
            tp += 1
        if p == t and t == 0:
            tn += 1
        if p != t and t == 1:
            fn += 1
        if p != t and t == 0:
            fp += 1
    return pred_labels, real_labels, tp, fp, tn, fn


def evaluate(model, validation_dataloader, device, epoch, steps, writer=EVAL_WRITER):
    LOGGER.info("======== Running Eval ========")
    model.eval()
    eval_steps = 0
    pred_labels, real_labels = [], []
    start = time.time()
    total_loss = 0.0
    tp = tn = fp = fn = 0
    for batch in validation_dataloader:
        d = tuple(t.to(device) for t in batch)
        pvs_input_ids, pvs_input_mask, pvs_token_ids, title_input_ids, title_input_mask, title_token_ids,  cate_input_ids, cate_input_mask, cate_token_ids, cate_path_input_ids, cate_path_input_mask, cate_path_token_ids, industry_name_input_ids, industry_name_input_mask, industry_name_token_ids, labels = d
        with torch.no_grad():
            output = model(pvs_input_ids=pvs_input_ids, pvs_token_type_ids=pvs_token_ids,
                           pvs_attention_mask=pvs_input_mask,
                           title_input_ids=title_input_ids, title_token_type_ids=title_token_ids,
                           title_attention_mask=title_input_mask,
                           cate_input_ids=cate_input_ids, cate_token_type_ids=cate_token_ids,
                           cate_attention_mask=cate_input_mask,
                           cate_path_input_ids=cate_path_input_ids, cate_path_token_type_ids=cate_path_token_ids,
                           cate_path_attention_mask=cate_path_input_mask,
                           industry_name_input_ids=industry_name_input_ids,
                           industry_name_token_type_ids=industry_name_token_ids,
                           industry_name_attention_mask=industry_name_input_mask,
                           next_sentence_label=labels)
            logits = output[1]
            total_loss += output[-1].item()
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            pred_labels_, real_labels_, tp_, fp_, tn_, fn_ = flat_preds_and_labels(logits, label_ids)
            tp += tp_
            tn += tn_
            fp += fp_
            fn += fn_
            pred_labels.extend(pred_labels_)
            real_labels.extend(real_labels_)
            eval_steps += 1
        t0 = time.time()
        if eval_steps % 100 == 0:
            LOGGER.info(f"Eval {eval_steps} batches, cost {t0 - start} 秒 ...")
            start = time.time()

    best_acc, best_f1, best_precision, best_recall, threshold = find_best_f1_and_threshold(pred_labels, real_labels,
                                                                                           True)
    cls_recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    cls_precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    cls_acc = (tp + tn) / (tp + tn + fp + fn)
    cls_F1 = (2 * cls_precision * cls_recall) / (cls_precision + cls_recall)
    writer.update(best_f1, best_precision, best_recall, best_acc, threshold,
                  cls_F1, cls_precision, cls_recall, cls_acc, 0.5,
                  epoch, steps)
    return best_f1, best_precision, best_recall, best_acc, \
           cls_F1, cls_precision, cls_recall, cls_acc, \
           total_loss / len(validation_dataloader)


def format_time(elapsed):
    '''Takes a time in seconds and returns a string hh:mm:ss'''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def save_model(tokenizer, model, output_dir):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    LOGGER.info(f"* Save model to {output_dir} ^_^.")


def save_checkpoint(model, optimizer, scheduler, global_steps, output_checkpoint):
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save({
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_steps": global_steps,
    },
        output_checkpoint)
    LOGGER.info(f"* Save checkpoint to {output_checkpoint} .")


def get_dataloader_from_files(data_dir, filename, batch_size, tokenizer, mode="train", shuffle_pvs=False):
    pvs, title, industry_name, cate, cate_path = join_data(data_dir, filename, do_shuffle=shuffle_pvs)
    show(pvs, mode, "pvs")
    show(title, mode, "title")
    show(industry_name, mode, "industry_name")
    show(cate, mode, "cate")
    show(cate_path, mode, "cate_path")

    pvs_src, pvs_tgt, titles_src, titles_tgt, cate_src, cate_tgt, cate_path_src, cate_path_tgt, \
    industry_name_src, industry_name_tgt, labels = get_examples(pvs, title, cate, cate_path, industry_name)

    show_pairs(pvs_src, pvs_tgt, labels, "pvs", mode=mode)
    show_pairs(titles_src, titles_tgt, labels, "title", mode=mode)
    show_pairs(cate_src, cate_tgt, labels, "cate", mode=mode)
    show_pairs(cate_path_src, cate_path_tgt, labels, "cate_path", mode=mode)
    show_pairs(industry_name_src, industry_name_tgt, labels, "industry_name", mode=mode)

    LOGGER.info(f"======== Encode {mode} data ==========")
    pvs, title, cate, cate_path, industry_name = encode(tokenizer,
                                                        pvs_src, pvs_tgt,
                                                        titles_src, titles_tgt,
                                                        cate_src, cate_tgt,
                                                        cate_path_src, cate_path_tgt,
                                                        industry_name_src, industry_name_tgt)

    dataloader = get_dataloader(pvs, title, cate, cate_path, industry_name, labels, batch_size=batch_size, mode=mode)
    return dataloader


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
    if torch.cuda.is_available():
        device = torch.device("cuda")
        LOGGER.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        LOGGER.info(f'We will use the GPU:{torch.cuda.get_device_name(0)}')
    else:
        LOGGER.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    OPT_NUM = 4
    epsilon = torch.tensor(0.05)
    _add_noise = True
    noise_name = "MIX"
    alpha_1 = 10 / 255
    alpha_2 = 16 / 255
    adversarial_train = False

    if adversarial_train:
        _add_noise = True
        LOGGER.info(f"Execute Adversarial Training OPT_NUM={OPT_NUM}, MODE={noise_name} .")
    else:
        _add_noise = False
        OPT_NUM = 1
        LOGGER.info(f"Execute Plain Training ^_^.")

    # =================== 训练/评估语料 ==================
    # 训练语料
    # base_dir = './'
    # data_dir = "/root/autodl-tmp/corp/"
    #     data_dir = "/Users/mengqy/competitions/commodity-alignment/corp/nsp"
    train_file = "item-align-train.json"
    val_file = "item-align-val.json"
    # =================== 训练/评估语料 ==================

    # =================== 模型参数 ===================
    #     model_name_or_path = 'bert-base-chinese'
    model_name_or_path = os.path.join(HOME, "Data/output/bert/bert_base")
#     model_name_or_path = '/root/autodl-tmp/EntityAlignNet/pretrain/PretrainBert'
    restore_dir = ""

    batch_size = 16
    num_epochs = 20
    patience_steps = 20000
    # =================== 模型参数 ===================

    # ========================= 保存路径设置 =======================
    # 相关参数保存路径
    # output_dir = os.path.join(base_dir, "bert-old")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # *根据最小eval loss存储模型
    best_loss_model_dir = os.path.join(OUTPUT_DIR, "LossModel")
    os.makedirs(best_loss_model_dir, exist_ok=True)
    # *根据最大 F1 存储模型
    best_F1_model_dir = os.path.join(OUTPUT_DIR, "F1Model")
    os.makedirs(best_F1_model_dir, exist_ok=True)
    output_checkpoint = os.path.join(OUTPUT_DIR, "checkpoint")
    resume_dir = output_checkpoint

    # 训练过程的一些性能指标的保存路径
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(logdir=LOG_DIR, comment="Linear")

    # 日志输出到文件
    log_name = f"train-{datetime.date.today()}"
    fh = logging.FileHandler(f"{LOG_DIR}/{log_name}.log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)
    # ========================= 保存路径设置 ========================

    LOGGER.info(
        f"parameters: num_epochs={num_epochs}, patience_steps={patience_steps}, restore_dir={restore_dir}, "
        f"resume_dir={resume_dir}, batch_size={batch_size}")

    # Load the BERT tokenizer.
    LOGGER.info(f'Loading tokenizer from {model_name_or_path} ^_^')
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
#     tokenizer.do_basic_tokenize = False
    LOGGER.info(f'Loading train data from {os.path.join(DATA_DIR, train_file)} ...')
    train_dataloader = get_dataloader_from_files(DATA_DIR, train_file, batch_size, tokenizer, mode="train",
                                                 shuffle_pvs=False)
    LOGGER.info(f'Get train dataloader successfully ^_^')

    LOGGER.info(f'Loading valid data from {os.path.join(DATA_DIR, val_file)} ...')
    eval_dataloader = get_dataloader_from_files(DATA_DIR, val_file, batch_size, tokenizer, mode="eval",
                                                shuffle_pvs=False)
    LOGGER.info(f'Get eval dataloader successfully ^_^')

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    best_f1 = 0.0
    global_steps = 0
    eval_avg_steps = check_steps = 1000
    best_eval_loss = 100000000.0
    log_steps = 100
    best_eval_loss_steps = 0
    best_F1_steps = 0
    total_loss = 0.0
    model = BertAlignModel.from_pretrained(model_name_or_path)
    if restore_dir != "":
        model = load_model(model, restore_dir)
    else:
        LOGGER.info("There is no restore model path.")
    model.to(device)

    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    LOGGER.info('The BERT model has {:} different named parameters.\n'.format(len(params)))
    LOGGER.info('==== Embedding Layer ====\n')
    for p in params[0:5]:
        LOGGER.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    LOGGER.info('==== First Transformer ====\n')
    for p in params[5:21]:
        LOGGER.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    LOGGER.info('==== Output Layer ====\n')
    for p in params[-4:]:
        LOGGER.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * num_epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=3000,
                                                num_training_steps=total_steps)

    if resume_dir != "" and os.path.exists(resume_dir):
        checkpoint = torch.load(resume_dir, map_location="cpu")
        new_dict = {}
        LOGGER.info(checkpoint.keys())
        for attr in checkpoint["model_state_dict"]:
            if attr.startswith("module."):
                new_dict[attr.replace("module.", "", 1)] = checkpoint[
                    "model_state_dict"
                ][attr]
                print("module:", attr)
            else:
                new_dict[attr] = checkpoint["model_state_dict"][attr]
        model.load_state_dict(new_dict)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        LOGGER.info("Load scheduler ...")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        LOGGER.info("Load optimizer ...")
        global_steps = checkpoint["global_steps"]
        del checkpoint
        LOGGER.info(f'Load checkpoint from {resume_dir} successfully, train continue ^_^ ...')
    else:
        LOGGER.info("There is no resume model path.")

    model.to(device)

    # ========================================
    #               Training
    # ========================================
    # 对抗训练模式
    if _add_noise and noise_name in ["FREE", "PGD", "MIX"]:
        title_delta = torch.zeros(batch_size, 150, 768)
        title_delta.requires_grad = True

        pvs_delta = torch.zeros(batch_size, 512, 768)
        pvs_delta.requires_grad = True

    writer.add_scalar("Eval/Acc", 0.0, global_steps)
    writer.add_scalar("Eval/Precision", 0.0, global_steps)
    writer.add_scalar("Eval/Recall", 0.0, global_steps)
    writer.add_scalar("Eval/F1", 0.0, global_steps)
    writer.add_scalar("Eval/cls_Acc", 0.0, global_steps)
    writer.add_scalar("Eval/cls_Precision", 0.0, global_steps)
    writer.add_scalar("Eval/cls_Recall", 0.0, global_steps)
    writer.add_scalar("Eval/cls_F1", 0.0, global_steps)
    t0 = time.time()
    start = t0
    improve = 'No optimization'
    for epoch_i in range(0, num_epochs):
        torch.cuda.empty_cache()
        LOGGER.info('======== Training Epoch {:} / {:} ========'.format(epoch_i + 1, num_epochs))
        model.train()
        for step, batch in enumerate(train_dataloader):
            d = tuple(t.to(device) for t in batch)
            pvs_input_ids, pvs_input_mask, pvs_token_ids, title_input_ids, title_input_mask, title_token_ids, cate_input_ids, cate_input_mask, cate_token_ids, cate_path_input_ids, cate_path_input_mask, cate_path_token_ids, industry_name_input_ids, industry_name_input_mask, industry_name_token_ids, labels = d

            for _ in range(OPT_NUM):
                if _add_noise:
                    pvs_noise = pvs_delta[:pvs_input_ids.size(0)].to(device)
                    title_noise = title_delta[:pvs_input_ids.size(0)].to(device)
                else:
                    if adversarial_train:
                        pvs_noise = None
                        title_noise = None
                        _add_noise = True
                    else:
                        pvs_noise = None
                        title_noise = None

                model.zero_grad()
                output = model(pvs_input_ids=pvs_input_ids, pvs_token_type_ids=pvs_token_ids,
                               pvs_attention_mask=pvs_input_mask,
                               title_input_ids=title_input_ids, title_token_type_ids=title_token_ids,
                               title_attention_mask=title_input_mask,
                               cate_input_ids=cate_input_ids, cate_token_type_ids=cate_token_ids,
                               cate_attention_mask=cate_input_mask,
                               cate_path_input_ids=cate_path_input_ids, cate_path_token_type_ids=cate_path_token_ids,
                               cate_path_attention_mask=cate_path_input_mask,
                               industry_name_input_ids=industry_name_input_ids,
                               industry_name_token_type_ids=industry_name_token_ids,
                               industry_name_attention_mask=industry_name_input_mask,
                               next_sentence_label=labels,
                               pvs_noise=pvs_noise, title_noise=title_noise)
                loss = output[-1]
                total_loss += loss.item()
                global_steps += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                rand = np.random.random()
                if _add_noise and (noise_name == "FREE" or (noise_name == "MIX" and rand < 0.15)):
                    pvs_grad = pvs_delta.grad.detach()
                    pvs_delta.data = pvs_delta + epsilon * torch.sign(pvs_grad)
                    pvs_delta.data[:pvs_input_ids.size(0)] = clamp(pvs_delta[:pvs_input_ids.size(0)], -epsilon, epsilon)
                    pvs_delta.grad.zero_()

                    title_grad = title_delta.grad.detach()
                    title_delta.data = title_delta + epsilon * torch.sign(title_grad)
                    title_delta.data[:pvs_input_ids.size(0)] = clamp(title_delta[:pvs_input_ids.size(0)], -epsilon,
                                                                     epsilon)
                    title_delta.grad.zero_()
                elif _add_noise and (noise_name == "PGD" or (noise_name == "MIX" and rand < 0.45)):
                    pvs_grad = pvs_delta.grad.detach()
                    pvs_delta.data.uniform_(-epsilon, epsilon)
                    pvs_delta.data = pvs_delta + alpha_1 * torch.sign(pvs_grad)
                    pvs_delta.data[:pvs_input_ids.size(0)] = clamp(pvs_delta[:pvs_input_ids.size(0)], -epsilon, epsilon)
                    pvs_delta.grad.zero_()

                    title_grad = title_delta.grad.detach()
                    title_delta.data.uniform_(-epsilon, epsilon)
                    title_delta.data = title_delta + alpha_1 * torch.sign(title_grad)
                    title_delta.data[:pvs_input_ids.size(0)] = clamp(title_delta[:pvs_input_ids.size(0)], -epsilon,
                                                                     epsilon)
                    title_delta.grad.zero_()
                else:
                    _add_noise = False

                if global_steps == 0:
                    writer.add_scalar("Train/Loss", total_loss, global_steps)

                if global_steps % log_steps == 0 and not global_steps == 0:
                    elapsed = format_time(time.time() - t0)
                    exhaust = format_time(time.time() - start)
                    writer.add_scalar("Train/Loss", total_loss / global_steps, global_steps)
                    LOGGER.info(
                        'Epoch {:>3,} - Steps {:>5,} - Total {:>5,} - Train/Loss {:>5.4} - Cost: {:} - Total cost: {:}.'.format(
                            epoch_i + 1,
                            global_steps,
                            len(train_dataloader),
                            loss.item(),
                            exhaust,
                            elapsed))
                    start = time.time()
                # ========================================
                #              Evaluation
                # ========================================
                if global_steps % eval_avg_steps == 0 and not global_steps == 0:
                    F1, pre, recall, acc, cls_F1, cls_pre, cls_recall, cls_acc, eval_loss = evaluate(model,
                                                                                                     eval_dataloader,
                                                                                                     device,
                                                                                                     epoch_i + 1,
                                                                                                     global_steps)
                    writer.add_scalar("Eval/Acc", acc, global_steps)
                    writer.add_scalar("Eval/Precision", pre, global_steps)
                    writer.add_scalar("Eval/Recall", recall, global_steps)
                    writer.add_scalar("Eval/F1", F1, global_steps)
                    writer.add_scalar("Eval/cls_Acc", cls_acc, global_steps)
                    writer.add_scalar("Eval/cls_Precision", cls_pre, global_steps)
                    writer.add_scalar("Eval/cls_Recall", cls_recall, global_steps)
                    writer.add_scalar("Eval/cls_F1", cls_F1, global_steps)
                    writer.add_scalar("Eval/Loss", eval_loss, global_steps)

                    LOGGER.info(
                        "* Epoch: {0:>3,}, Steps {1:>5,}, Accuracy: {2:>5.4}, Precision: {3:>5.4}, Recall: {4:>5.4}, F1: {5:.>5.4}, Eval/Loss: {6:.>5.4}, Time cost: {7:}".format(
                            epoch_i + 1, global_steps, acc, pre, recall, F1, eval_loss, format_time(time.time() - t0)))

                    LOGGER.info(
                        "* CLS Epoch: {0:>3,}, Steps {1:>5,}, Accuracy: {2:>5.4}, Precision: {3:>5.4}, Recall: {4:>5.4}, F1: {5:.>5.4}, Eval/Loss: {6:.>5.4}, Time cost: {7:}".format(
                            epoch_i + 1, global_steps, cls_acc, cls_pre, cls_recall, cls_F1, eval_loss,
                            format_time(time.time() - t0)))

                    if best_eval_loss > eval_loss or best_f1 < F1:
                        if best_eval_loss > eval_loss:
                            best_eval_loss = eval_loss
                            best_eval_loss_steps = global_steps
                            save_model(tokenizer, model, best_loss_model_dir)
                            improve = "Optimize Eval/Loss ^_^"
                            LOGGER.info(f"***********************************")
                            LOGGER.info(
                                f"* Current Best Eval/Loss {best_eval_loss} epoch:{epoch_i + 1}, steps:{global_steps} ^_^")
                            LOGGER.info(f"***********************************")
                        if best_f1 < F1:
                            best_f1 = F1
                            best_F1_steps = global_steps
                            save_model(tokenizer, model, best_F1_model_dir)
                            if improve != "No optimization":
                                improve += " and F1 ^_^"
                            else:
                                improve = "Optimize F1 ^_^"
                            LOGGER.info(f"***********************************")
                            LOGGER.info(
                                f"* Current Best F1 {best_f1} epoch:{epoch_i + 1}, steps:{global_steps} Eval/Loss:{eval_loss} ^_^")
                            LOGGER.info(f"***********************************")

                    if global_steps % check_steps == 0:
                        save_checkpoint(model, optimizer, scheduler, global_steps, output_checkpoint)

                    if global_steps - best_eval_loss_steps >= patience_steps and global_steps - best_F1_steps >= patience_steps:
                        LOGGER.info(
                            f"There is no optimizing model after {max(global_steps - best_eval_loss_steps, global_steps - best_F1_steps)} steps, Early stopping ....")
                        LOGGER.info(f"***********************************")
                        LOGGER.info(
                            f"* Best F1 {best_f1}  after {epoch_i + 1} epochs, {global_steps} steps training ^_^.")
                        LOGGER.info(f"***********************************")
                        LOGGER.info("Training successfully ^_^.")
                        return
                    model.train()
                    start = time.time()
                    LOGGER.info(f"{improve}, continue training ...")
                    improve = 'No optimization'


if __name__ == "__main__":
    main()



