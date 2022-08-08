# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :EntityAlignNet
# @File     :run_bert_pretrain
# @Date     :2022/7/5 16:58
# @Author   :mengqingyang
# @Email    :mengqingyang0102@163.com
-------------------------------------------------
"""
import random
from copy import deepcopy
from typing import List, Tuple
import json
import numpy as np
import torch
import os
from pytorch_transformers import BertTokenizer, AdamW, BertForPreTraining
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import get_linear_schedule_with_warmup

from .log import LOGGER


if torch.cuda.is_available():
    device = torch.device("cuda")
    LOGGER.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
    LOGGER.info(f'We will use the GPU:{torch.cuda.get_device_name(0)}')
else:
    LOGGER.info('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def read_pretrained_data(data_dir, file, batch_size=1000):
    records = []
    file_name = os.path.join(data_dir, file)
    with open(file_name, 'r', encoding="utf8") as reader:
        for rec in reader:
            records.append(json.loads(rec.strip()))
            if len(records) == batch_size:
                random.shuffle(records)
                yield records
                records = []
    if records:
        random.shuffle(records)
        yield records


def join_pretraining_data(record, field_names: List[str]):
    return {field: record.get(field, "信息缺失") for field in field_names}


def truncate_tokens(tokens, seq_len):
    if seq_len == -1:
        return tokens
    if len(tokens) < seq_len:
        return tokens
    rand = np.random.random()
    if rand < 0.5:
        return tokens[:seq_len]
    else:
        return tokens[-seq_len:]


def truncate_pairs(tokens, label_ids, seq_len):
    if seq_len == -1:
        return tokens, label_ids
    if len(tokens) < seq_len:
        return tokens, label_ids
    rand = np.random.random()
    if rand < 0.5:
        return tokens[:seq_len], label_ids[:seq_len]
    else:
        return tokens[-seq_len:], label_ids[-seq_len:]


def create_input_features(examples, max_seq_len, tokenizer, next_label=1):
    org_tokens = []
    input_ids = []
    token_type_ids = []
    attention_mask = []
    label_ids = []

    for exp in examples[:-1]:
        org_tokens.extend(exp["org_tokens"])
        input_ids.extend(exp["input_ids"])
        token_type_ids.extend(exp["token_type_ids"])
        attention_mask.extend(exp["attention_mask"])
        label_ids.extend(exp["label_ids"])
    total_len = len(input_ids)
    if total_len >= max_seq_len:
        return []
    # truncate pairs
    pvs_input_ids = examples[-1]["input_ids"]
    pvs_label_ids = examples[-1]["label_ids"]
    # print(max_seq_len - total_len, len(pvs_label_ids))
    pvs_input_ids, pvs_label_ids = truncate_pairs(pvs_input_ids, pvs_label_ids, max_seq_len - total_len)

    # add
    input_ids.extend(pvs_input_ids)
    label_ids.extend(pvs_label_ids)
    attention_mask.extend([1] * len(pvs_input_ids))
    token_type_ids.extend([4] * len(pvs_input_ids))

    # add [CLS] and [SEP]
    input_ids = [tokenizer.convert_tokens_to_ids("[CLS]")] + input_ids + [tokenizer.convert_tokens_to_ids("[SEP]")]
    label_ids = [-1] + label_ids + [-1]
    token_type_ids = [0] + token_type_ids + [0]
    attention_mask = [0] + attention_mask + [0]

    while len(input_ids) < max_seq_len + 2:
        input_ids.append(0)
        attention_mask.append(0)
        token_type_ids.append(0)
        label_ids.append(-1)

    assert len(input_ids) == max_seq_len + 2
    assert len(attention_mask) == max_seq_len + 2
    assert len(token_type_ids) == max_seq_len + 2
    assert len(label_ids) == max_seq_len + 2

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "label_ids": label_ids,
        "next_label": next_label
    }


def get_next_examples(seqs: List[str], tokenizer, max_seq_len, data, field_names, neg_sample_num=100):
    next_examples = []
    for i in range(neg_sample_num):
        rint = random.randint(0, len(data) - 1)
        d = join_pretraining_data(data[rint], field_names=field_names)
        assert len(d) == len(field_names)
        d["item_pvs"] = d["item_pvs"].replace('#', '')
        d = [d[field] for field in field_names]
        seqs[-1] = d[-1]
        examples = []
        for idx, seq in enumerate(seqs):
            org_tokens = tokenizer.tokenize(seq)
            input_ids = tokenizer.convert_tokens_to_ids(org_tokens)
            assert len(input_ids) == len(org_tokens)
            attention_mask = [1] * len(input_ids)
            token_type_ids = [idx] * len(input_ids)
            label_ids = [-1] * len(input_ids)
            examples.append({
                "org_tokens": org_tokens,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "label_ids": label_ids
            })
        next_examples.append(create_input_features(examples, max_seq_len, tokenizer, 0))
    return next_examples


# industry_name, cate_name, cate_name_path, title, item_pvs
def get_masked_examples(seqs: List[str], tokenizer, max_seq_len) -> Tuple[List[dict], List[dict]]:
    org_examples, masked_examples = [], []
    for idx, seq in enumerate(seqs):
        org_tokens = tokenizer.tokenize(seq)
        input_ids = tokenizer.convert_tokens_to_ids(org_tokens)
        assert len(input_ids) == len(org_tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [idx] * len(input_ids)
        label_ids = [-1] * len(input_ids)
        org_examples.append({
            "org_tokens": org_tokens,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label_ids": label_ids
        })

    title_tokens = org_examples[-2]["org_tokens"]
    pv_tokens = org_examples[-1]["org_tokens"]
    props, title_masked_positions = process_title_match_pvs(title_tokens, pv_tokens)

    # industry mask.
    industry_mask_examples = [deepcopy(exp) for exp in org_examples]
    input_ids, label_ids = do_whole_mask(industry_mask_examples[0]["input_ids"], tokenizer)
    industry_mask_examples[0]["input_ids"] = input_ids
    industry_mask_examples[0]["label_ids"] = label_ids
    masked_examples.append(create_input_features(industry_mask_examples, max_seq_len, tokenizer))

    # cate name mask.
    cate_mask_examples = [deepcopy(exp) for exp in org_examples]
    input_ids, label_ids = do_whole_mask(cate_mask_examples[1]["input_ids"], tokenizer)
    cate_mask_examples[1]["input_ids"] = input_ids
    cate_mask_examples[1]["label_ids"] = label_ids
    masked_examples.append(create_input_features(cate_mask_examples, max_seq_len, tokenizer))

    # title mask.
    title_mask_examples = [deepcopy(exp) for exp in org_examples]
    input_ids, label_ids = do_title_mask(title_mask_examples[3]["input_ids"], title_masked_positions, tokenizer)
    title_mask_examples[3]["input_ids"] = input_ids
    title_mask_examples[3]["label_ids"] = label_ids
    masked_examples.append(create_input_features(title_mask_examples, max_seq_len, tokenizer))

    # pvs mask.
    pvs_mask_features = do_pvs_mask(props, tokenizer)
    for pmf in pvs_mask_features:
        pvs_mask_examples = [deepcopy(exp) for exp in org_examples]
        pvs_mask_examples[4] = pmf
        masked_examples.append((create_input_features(pvs_mask_examples, max_seq_len, tokenizer)))

    return org_examples, masked_examples


def do_pvs_mask(props, tokenizer):
    tokens = []
    masked_prop_key_positions = []
    masked_prop_value_positions = []
    for prop in props:
        masked_prop_key_positions.append([len(tokens), len(tokens) + len(prop[0])])
        tokens.extend(prop[0] + [":"])
        masked_prop_value_positions.append([len(tokens), len(tokens) + len(prop[1])])
        tokens.extend(prop[1] + [";"])
    masked_positions = masked_prop_value_positions + masked_prop_key_positions
    np.random.shuffle(masked_positions)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # input_ids = tokens
    examples = []
    # print("=================================================")
    for masked_pos in masked_positions:
        input_ids_ = [d for d in input_ids]
        label_ids = [-1] * len(tokens)
        rand = np.random.random()
        if rand < 0.8:
            # 不做任何操作
            pass
        elif rand < 0.9:
            # 随机替换
            for p in range(masked_pos[0], masked_pos[1]):
                input_ids_[p] = np.random.randint(len(tokenizer))
                # input_ids_[p] = '#####'
        else:
            #
            for p in range(masked_pos[0], masked_pos[1]):
                input_ids_[p] = tokenizer.convert_tokens_to_ids("[MASK]")
                # input_ids_[p] = "[MASK]"
        for p in range(masked_pos[0], masked_pos[1]):
            label_ids[p] = input_ids[p]
        # print(input_ids_)
        # print(label_ids)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [4] * len(input_ids)

        # ts = tokenizer.convert_ids_to_tokens(input_ids_)
        # ls = [tokenizer.convert_ids_to_tokens(l) for l in label_ids if l != -1]
        # print(ts)
        # print("labels:", ls)
        examples.append({
            "org_tokens": tokens,
            "input_ids": input_ids_,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label_ids": label_ids
        })
        # examples.append([input_ids_, label_ids])
    return examples


def do_title_mask(tokens, mask_positions, tokenizer):
    label_ids = [-1] * len(tokens)
    if not mask_positions:
        for i in range(len(label_ids)):
            rand = np.random.random()
            if rand < 0.15:
                label_ids[i] = tokens[i]
        return tokens, label_ids
    rand = np.random.random()
    for pos in mask_positions:
        if rand < 0.5:
            for i in range(pos[0], pos[1]):
                label_ids[i] = tokens[i]
                tokens[i] = np.random.randint(len(tokenizer))
        else:
            for i in range(pos[0], pos[1]):
                label_ids[i] = tokens[i]
                tokens[i] = tokenizer.convert_tokens_to_ids("[MASK]")
    return tokens, label_ids


def do_whole_mask(tokens, tokenizer):
    rand = np.random.random()
    if rand < 0.8:
        return tokens, [t for t in tokens]
    elif rand < 0.9:
        label_ids = [t for t in tokens]
        for i in range(len(tokens)):
            tokens[i] = np.random.randint(len(tokenizer))
    else:
        label_ids = [t for t in tokens]
        tokens = [tokenizer.convert_tokens_to_ids("[MASK]")] * len(tokens)
    return tokens, label_ids


def process_single_property(chunk):
    if not chunk or ":" not in chunk:
        return []
    sep_pos = chunk.index(":")
    p = chunk[0: sep_pos]
    v = chunk[sep_pos + 1:]
    if v and v[-1] == ";": v = v[:-1]
    return [p, v]


def do_match_terms(title, v):
    mask_positions = []
    for i in range(len(title)):
        if ''.join(title[i: i + len(v)]) == ''.join(v):
            # print(''.join(title[i: i + len(v)]), ''.join(title))
            mask_positions.append((i, i + len(v)))
    return mask_positions


def process_title_match_pvs(title, pvs):
    chunks = []
    end = -1
    for idx, token in enumerate(pvs):
        if token == ";" or idx == len(pvs) - 1:
            chunks.append(pvs[end + 1: idx + 1])
            end = idx
    props = []
    if not chunks:
        return props, []
    else:
        t = 0
        for c in chunks:
            t += len(c)
            res = process_single_property(c)
            if res:
                props.append(res)
        assert pvs and (len(pvs) == t), f"======{pvs} == {t}"

    t = 0
    for prop in props:
        t += len(prop[0])
        t += len(prop[1])
        t += 2

    title_masked_positions = []
    for prop in props:
        title_masked_positions.extend(do_match_terms(title, prop[1]))
    return props, title_masked_positions


def process_pvs_mask(props, tokenizer):
    tokens = []
    # 属性名称 属性取值 tokenize
    for p in props:
        p_tokens = tokenizer.tokenize(p["prop_name"] + ":")
        v_tokens = tokenizer.tokenize(p["prop_value"] + ";")
        tokens.append(tokenizer.convert_tokens_to_ids(p_tokens))
        tokens.append(tokenizer.convert_tokens_to_ids(v_tokens))

    # mask策略 0.8比例保持不变, 0.1随机化, 0.1[MASK]替代.
    masked_tokens = []
    for idx, token in enumerate(tokens):
        rand = np.random.random()
        if rand < 0.8:
            # 保持不变
            masked_tokens.append((idx, token))
        elif rand < 0.9:
            # [MASK]代替
            masked_tokens.append((idx, [tokenizer.convert_tokens_to_ids("[MASK]")] * len(token[:-1]) + [tokens[-1]]))
        else:
            # 随机化替换
            rand_tokens = [0] * len(token[:-1])
            for i in range(len(token[:-1])):
                rand_tokens[i] = np.random.randint(len(tokenizer))
            masked_tokens.append((idx, rand_tokens + [tokens[-1]]))

    masked_pvs_tokens = []
    masked_label_ids = []
    for m in masked_tokens:
        if m[0] == 0:
            r = [a for t in tokens[1:] for a in t]
            masked_label_ids.append(tokens[0][:-1] + (len(r) + 1) * [-1])
            masked_pvs_tokens.append(m[1] + r)
        elif m[0] == len(tokens) - 1:
            l = [a for t in tokens[:-1] for a in t]
            masked_pvs_tokens.append(l + m[1])
            masked_label_ids.append([-1] * len(l) + tokens[m[0]][:-1] + [-1])
        else:
            l = [a for t in tokens[:m[0]] for a in t]
            r = [a for t in tokens[m[0] + 1:] for a in t]
            masked_label_ids.append(len(l) * [-1] + tokens[m[0]][:-1] + (len(r) + 1) * [-1])
            masked_pvs_tokens.append(l + m[1] + r)

    for mpt, mli in zip(masked_pvs_tokens, masked_label_ids):
        assert len(mpt) == len(mli)

    return masked_pvs_tokens, masked_label_ids


def get_pretrain_dataloader(examples, batch_size=8):
    input_ids, attention_mask, token_type_ids, label_ids, next_labels = [], [], [], [], []
    for exp in examples:
        input_ids.append(exp["input_ids"])
        attention_mask.append(exp["attention_mask"])
        token_type_ids.append(exp["token_type_ids"])
        label_ids.append(exp["label_ids"])
        next_labels.append(exp["next_label"])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    label_ids = torch.tensor(label_ids, dtype=torch.long)
    next_labels = torch.tensor(next_labels, dtype=torch.long)

    # Create the DataLoader.
    data = TensorDataset(input_ids, attention_mask, token_type_ids, label_ids, next_labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader


def get_total_steps(data_dir, filename):
    total_len = 0
    for data in read_pretrained_data(data_dir, filename):
        total_len += len(data)
    return total_len


def get_eval_dataloader(data_dir, filename, field_names, tokenizer, max_seq_len, batch_size):
    eval_data = None
    for data in read_pretrained_data(data_dir, filename, -1):
        eval_data = data
    bunch_examples = []
    for record in eval_data:
        d = join_pretraining_data(record, field_names=field_names)
        assert len(d) == len(field_names)
        d["item_pvs"] = d["item_pvs"].replace('#', '')
        d = [d[field] for field in field_names]
        _, masked_examples = get_masked_examples(d, tokenizer, max_seq_len)
        next_examples = get_next_examples(d, tokenizer, max_seq_len, eval_data, field_names)
        bunch_examples.extend(masked_examples)
#         bunch_examples.extend(next_examples)
        np.random.shuffle(bunch_examples)
    return get_pretrain_dataloader(bunch_examples[:], batch_size=batch_size)


def eval_model(model, dataloader, epoch_idx, steps):
    LOGGER.info(f"====== Eval Model {epoch_idx}/{steps}=======")
    total_loss = 0.0
    for step, batch in enumerate(dataloader):
        batch = [b.to(device) for b in batch]
        input_ids, attention_mask, token_type_ids, label_ids, next_labels = batch
        with torch.no_grad():
            output = model(input_ids,
                           token_type_ids=None,
                           attention_mask=attention_mask,
                           masked_lm_labels=label_ids,
                           next_sentence_label=next_labels)
            total_loss += output[0].item()
        if step % 100 == 0:
            LOGGER.info(f"Eval {step} batches ...")
    return total_loss / len(dataloader)


def save_model(tokenizer, model, output_dir):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    # data_dir = "/Users/mengqy/competitions/commodity-alignment/corp"
    base_dir = "./"
    data_dir = "/root/autodl-tmp/pretrain-corp"
    train_file = "pretrain_train.jsonl"
    valid_file = "pretrain_valid.jsonl"
    # transformers.logging.set_verbosity(transformers.logging.ERROR)
    max_seq_len = 510
    batch_size = 32
    num_epochs = 20
    model_name_or_path = "bert-base-chinese"
#     model_name_or_path = "/root/autodl-tmp/roberta"
    # model_name_or_path = "./albert_base/"
    field_names = ["industry_name", "cate_name", "cate_name_path", "title", "item_pvs"]
    log_steps = 100
    eval_steps = 2000
    patience_steps = 20000

    output_dir = "PretrainBert"
    output_dir = os.path.join(base_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_name = "pretrain-bert"
    fh = logging.FileHandler(f"{output_dir}/{log_name}.log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)

#     log_dir = os.path.join(output_dir, "roberta-pretrain")
    log_dir = os.path.join("/root/tf-logs", "PretrainBert")
    writer = SummaryWriter(logdir=log_dir, comment="Linear")

    data_size = get_total_steps(data_dir, train_file)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    LOGGER.info('Loading BERT tokenizer ^_^')
    model = BertForPreTraining.from_pretrained(model_name_or_path)
    LOGGER.info('Loading BERT model params ^_^')
    model.to(device)
    # 查看载入的模型参数
    params = list(model.named_parameters())
    LOGGER.info('The BERT model has {:} different named parameters.\n'.format(len(params)))
    LOGGER.info('==== Embedding Layer ====\n')
    for p in params[0:5]:
        LOGGER.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    LOGGER.info('\n==== First Transformer ====\n')
    for p in params[5:25]:
        LOGGER.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    LOGGER.info('\n==== Output Layer ====\n')
    for p in params[-4:]:
        LOGGER.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = data_size // batch_size * num_epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=3000,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    eval_dataloader = get_eval_dataloader(data_dir, valid_file,
                                          field_names, tokenizer,
                                          max_seq_len, batch_size)
    LOGGER.info("Loading Eval data successfully ^_^")

    global_steps = 0
    best_eval_loss = 100000000000
    best_steps = 0

    for epoch_idx in range(num_epochs):
        model.train()
        for data in read_pretrained_data(data_dir, train_file):
            bunch_examples = []
            for record in data:
                d = join_pretraining_data(record, field_names=field_names)
                assert len(d) == len(field_names)
                d["item_pvs"] = d["item_pvs"].replace('#', '')
                d = [d[field] for field in field_names]
                _, masked_examples = get_masked_examples(d, tokenizer, max_seq_len)
                next_examples = get_next_examples(d, tokenizer, max_seq_len, data, field_names)
                bunch_examples.extend(masked_examples)
#                 bunch_examples.extend(next_examples)
            random.shuffle(bunch_examples)
            data_loader = get_pretrain_dataloader(bunch_examples, batch_size=batch_size)
            for step, batch in enumerate(data_loader):
                batch = [b.to(device) for b in batch]
                input_ids, attention_mask, token_type_ids, label_ids, next_labels = batch
                # print("input_ids.shape:", input_ids.shape)
                # print("attention_mask.shape:", attention_mask.shape)
                # print("token_type_ids.shape:", token_type_ids.shape)
                # print("label_ids.shape:", label_ids.shape)
                model.zero_grad()
                output = model(input_ids,
                               token_type_ids=None,
                               attention_mask=attention_mask,
                               masked_lm_labels=label_ids,
                               next_sentence_label=next_labels)
                loss = output[0]
                loss.backward()
                global_steps += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                if global_steps != 0 and global_steps % log_steps == 0:
                    writer.add_scalar("Train/loss", loss.item(), global_steps)
                    LOGGER.info(
                        'Epoch {:>3,}, Steps {:>5,}, Total {:>5,}, Train/Loss {:>5.4}'.format(epoch_idx + 1,
                                                                                              global_steps,
                                                                                              total_steps,
                                                                                              loss.item()))

                if global_steps != 0 and global_steps % eval_steps == 0:
                    eval_loss = eval_model(model, eval_dataloader, epoch_idx, global_steps)
                    writer.add_scalar("Eval/loss", eval_loss, global_steps)
                    improve = '  No optimization! ...'
                    if best_eval_loss > eval_loss:
                        best_eval_loss = eval_loss
                        best_steps = global_steps
                        improve = ' ^_^'
                        save_model(tokenizer, model, output_dir)
                    else:
                        if global_steps - best_steps > patience_steps:
                            LOGGER.info(f"Eval loss has not been optimized for {patience_steps} steps, Early stopping ...")
                            LOGGER.info(f"Pretraining successfully, num_epochs:{epoch_idx+1}, global_steps:{global_steps} ^_^")
                            return
                    LOGGER.info(
                        '* Epoch {:>3,}, Steps {:>5,}, Total {:>5,}, Eval/Loss {:>5.4}'.format(epoch_idx + 1,
                                                                                             global_steps,
                                                                                             total_steps,
                                                                                             eval_loss) + improve)


if __name__ == "__main__":
    main()

