import os
import random
import argparse
import json
import numpy as np
import torch

from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertConfig,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import precision_score, recall_score, f1_score
from src.models import GCNTwoTower
from src.data import GCNDataset, collate_gnn
from src.utils import logger, BOS_TOKEN


def get_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", required=True, type=str, help="模型训练数据地址")
    parser.add_argument("--output_dir", required=True, type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--config_file", required=True, type=str, help="The config file which specified the model details.")
    parser.add_argument("--model_name", required=True, type=str, help="model saving name",)
    parser.add_argument("--data_version", required=True, type=str, help="data version")
    parser.add_argument("--interaction_type", required=True, type=str,
                        help="交互方式, one_tower: 中间过程有交互, two_tower: 中间过程无交互，最后embedding交互",)
    parser.add_argument("--classification_method", required=True, type=str,
                        help="分类方法, cls: 使用cls作为分类符, vec_sim: 计算2个item的vector，通过vector similarity和阈值来分类")
    parser.add_argument("--similarity_measure", required=True, type=str,
                        help="向量相似度量: cosine, inner_product, l1 (l1 euclidean distance), l2 (l2 euclidean distance)",)
    parser.add_argument("--loss_type", required=True, type=str,
                        help="损失函数类型, ce: cross entropy, bce: binary cross entropy with logits loss, cosine: cosine embedding loss",)
    # training
    parser.add_argument("--do_train", action="store_true", help="是否进行模型训练")
    parser.add_argument("--do_eval", action="store_true", help="是否进行模型验证")
    parser.add_argument("--do_pred", action="store_true", help="是否进行模型测试")
    parser.add_argument("--seed", default=2345, type=int, help="random seed")
    parser.add_argument("--train_batch_size", default=512, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=1024, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--start_epoch", default=0, type=int, help="starting training epoch")
    parser.add_argument("--num_train_epochs", default=500, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="weight decay")
    parser.add_argument("--log_steps", default=None, type=int, help="every n steps, log training process")
    parser.add_argument("--save_epochs", default=10, type=int, help="every n epochs, save model and eval")
    parser.add_argument("--pretrained_model_path", default=None, type=str, help="pretrained model path, including roberta and pkgm")
    parser.add_argument("--file_state_dict", default=None, type=str, help="finetuned model path")
    parser.add_argument("--parameters_to_freeze", default=None, type=str, help="file that contains parameters that do not require gradient descend")
    parser.add_argument("--threshold", default=0.5, type=float, help="default threshold for item embedding score for prediction")
    # optimization
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for. "
                                                                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--margin", default=1.0, type=float, help="margin in loss function")
    # NLP
    parser.add_argument("--do_lower_case", default=True, type=bool, help="Whether to lower case the input text. True for uncased models, False for cased models.")
    # GNN
    parser.add_argument("--num_layers", default=4, type=int, help="number of gcn layers")
    parser.add_argument("--hidden_size", default=128, type=int, help="gcn hidden_size")
    parser.add_argument("--feature_dim", default=1024, type=int, help="feature matrix dim (equal to roberta large hidden size)")
    parser.add_argument("--alpha", default=0.1, type=float, help="gcn layer param")
    parser.add_argument("--theta", default=0.5, type=float, help="gcn layer param")

    return parser.parse_args()


def load_raw_data(args):
    f = os.path.join(args.data_dir, "processed", "entity2id.txt")
    e2id = dict()
    with open(f, "r", encoding="utf-8") as r:
        while True:
            line = r.readline()
            if not line:
                break
            k, v = line.strip("\n").split("\t")
            if "/item/" not in k:
                continue
            item_id = k.replace("/item/", "")
            e2id[item_id] = int(v)

    f = os.path.join(args.data_dir, "raw", "item_train_train_pair.jsonl")
    train_data = []
    with open(f, "r", encoding="utf-8") as r:
        while True:
            line = r.readline()
            if not line:
                break
            d = json.loads(line.strip("\n"))
            d['src_idx'] = e2id[d['src_item_id']]
            d['tgt_idx'] = e2id[d['tgt_item_id']]
            train_data.append(d)

    f = os.path.join(args.data_dir, "raw", "item_train_valid_pair.jsonl")
    valid_data = []
    with open(f, "r", encoding="utf-8") as r:
        while True:
            line = r.readline()
            if not line:
                break
            d = json.loads(line.strip("\n"))
            d['src_idx'] = e2id[d['src_item_id']]
            d['tgt_idx'] = e2id[d['tgt_item_id']]
            valid_data.append(d)

    f = os.path.join(args.data_dir, "raw", "item_valid_pair.jsonl")
    test_data = []
    with open(f, "r", encoding="utf-8") as r:
        while True:
            line = r.readline()
            if not line:
                break
            d = json.loads(line.strip("\n"))
            d['src_idx'] = e2id[d['src_item_id']]
            d['tgt_idx'] = e2id[d['tgt_item_id']]
            test_data.append(d)

    return train_data, valid_data, test_data


def main():
    args = get_parser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpu = torch.cuda.device_count()
    logger.info(f"device: {device}, n_gpu: {n_gpu}, 16-bits training: {args.fp16}")
    # 设定随机数种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path, do_lower_case=args.do_lower_case)
    tokenizer.do_basic_tokenize = False
    tokenizer.bos_token = BOS_TOKEN
    logger.info(f"vocab size: {tokenizer.vocab_size}")
    # load model
    config = BertConfig.from_json_file(os.path.join(args.output_dir, args.config_file))
    config.interaction_type = args.interaction_type
    config.classification_method = args.classification_method
    config.similarity_measure = args.similarity_measure
    config.loss_type = args.loss_type
    config.intermediate_size = args.hidden_size
    config.num_hidden_layers = args.num_layers
    config.hidden_size = args.feature_dim
    config.alpha = args.alpha
    config.theta = args.theta
    if "gcn" in args.model_name:
        model = GCNTwoTower(config=config)
    # elif "graph_sage" in args.model_name:
    #     if args.interaction_type == "one_tower":
    #         model = RobertaOneTower.from_pretrained(args.pretrained_model_path, config=config,
    #                                                 ignore_mismatched_sizes=True)
    #     elif args.interaction_type == "two_tower":
    #         model = RobertaTwoTower.from_pretrained(args.pretrained_model_path, config=config,
    #                                                 ignore_mismatched_sizes=True)
    #     else:
    #         raise ValueError("interaction type should be: one_tower or two_tower")
    else:
        raise ValueError("model name should be: roberta or pkgm")
    # 冻结部分模型参数
    if args.parameters_to_freeze is not None:
        parameters_to_freeze = json.load(open(args.parameters_to_freeze, "r", encoding="utf-8"))
        parameters_freezed = []
        # for name in parameters_to_freeze:
        #     if "embeddings" in name:
        #         bert_weight_name_filtered.append(name)
        #     elif "encoder" in name:
        #         layer_num = name.split(".")[2]
        #         if int(layer_num) <= args.freeze:
        #             bert_weight_name_filtered.append(name)
        for key, value in dict(model.named_parameters()).items():
            if key.replace("roberta.", "") in parameters_to_freeze:
                parameters_freezed.append(key)
                value.requires_grad = False
        logger.info(f"Parameters to freeze: {parameters_to_freeze}")
        logger.info(f"Parameters freezed: {parameters_freezed}")
    # load previous model weights (if exists)
    if args.file_state_dict is not None:
        state_dict = torch.load(args.file_state_dict, map_location="cpu")
        model.load_state_dict(state_dict)
    # load raw data
    train_data, valid_data, test_data = load_raw_data(args)
    logger.info(f"# train samples: {len(train_data)}, # valid samples: {len(valid_data)}, # test samples: {len(test_data)}")
    # load feature matrix and adjacency matrix
    f = os.path.join(args.data_dir, "processed", "adj_t.pt")
    edge_index = torch.load(f, map_location="cpu").to(device=device, dtype=torch.float32)
    f = os.path.join(args.data_dir, "processed", "feature_matrix.pt")
    feature_matrix = torch.load(f, map_location="cpu").to(device=device, dtype=torch.float32)

    if args.do_train:
        if "gcn" in args.model_name:
            train_dataset = GCNDataset(train_data)
            train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                           collate_fn=collate_gnn)
        # elif "graph_sage" in args.model_name:
        #     if args.interaction_type == "one_tower":
        #         train_dataset = RobertaOneTowerDataset(train_data, tokenizer, max_seq_len=args.max_seq_len,
        #                                                max_seq_len_pv=args.max_seq_len_pv, classification_method=args.classification_method,
        #                                                auxiliary_task=args.auxiliary_task)
        #         train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
        #                                        collate_fn=collate_one_tower)
        #     elif args.interaction_type == "two_tower":
        #         train_dataset = RobertaTwoTowerDataset(train_data, tokenizer, max_seq_en=args.max_seq_len,
        #                                                max_seq_len_pv=args.max_seq_len_pv)
        #         train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
        #                                        collate_fn=collate_two_tower)
        #     else:
        #         raise ValueError("interaction type should be: one_tower or two_tower")
        else:
            raise ValueError("model name should be: roberta or pkgm")
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                                      betas=(0.9, 0.98))
        num_train_optimization_steps = int(
            len(train_dataset)
            / args.train_batch_size
            / args.gradient_accumulation_steps
        ) * (args.num_train_epochs - args.start_epoch)
        num_warmup_steps = int(num_train_optimization_steps * args.warmup_proportion)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_optimization_steps)

    if args.do_eval:
        if "gcn" in args.model_name:
            valid_dataset = GCNDataset(valid_data)
            valid_data_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                           collate_fn=collate_gnn)
        # elif "graph_sage" in args.model_name:
        #     if args.interaction_type == "one_tower":
        #         valid_dataset = RobertaOneTowerDataset(valid_data, tokenizer, max_seq_len=args.max_seq_len,
        #                                                max_seq_len_pv=args.max_seq_len_pv, classification_method=args.classification_method,
        #                                                auxiliary_task=args.auxiliary_task)
        #         valid_data_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False,
        #                                        collate_fn=collate_one_tower)
        #     elif args.interaction_type == "two_tower":
        #         valid_dataset = RobertaTwoTowerDataset(valid_data, tokenizer, max_seq_en=args.max_seq_len,
        #                                                max_seq_len_pv=args.max_seq_len_pv)
        #         valid_data_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False,
        #                                        collate_fn=collate_two_tower)
        #     else:
        #         raise ValueError("interaction type should be: one_tower or two_tower")
        else:
            raise ValueError("model name should be: roberta or pkgm")

    if device == "cuda":
        model.cuda()
        if args.do_train:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

    model_path = f"{args.model_name}-{args.data_version}-{args.interaction_type}-{args.classification_method}-{args.similarity_measure}-{args.loss_type}"
    output_model_path = os.path.join(args.output_dir, model_path)
    if not os.path.exists(output_model_path):
        os.mkdir(output_model_path)

    if args.do_train:
        # save all the hidden parameters.
        with open(os.path.join(output_model_path, "hyperparamter.txt"), "w") as f:
            print(args, file=f)  # Python 3.x
            print("\n", file=f)
            print(config, file=f)

        if args.fp16:
            scaler = torch.cuda.amp.GradScaler()

        logger.info("***** Running training *****")
        logger.info("  Model name = %s", model_path)
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        logger.info("  Learning rate = %.5f", args.learning_rate)

        global_step = 0
        for epoch in range(int(args.start_epoch), int(args.num_train_epochs)):
            model.train()
            for step, batch in enumerate(train_data_loader):
                optimizer.zero_grad()
                if args.fp16:
                    with torch.cuda.amp.autocast():
                        if args.interaction_type == "one_tower":
                            pair_indices = [t.to(device=device, non_blocking=True) for t in batch[2]]
                            batch = tuple(t.to(device=device, non_blocking=True) if t is not None else t for t in batch[3:])
                            input_ids, segment_ids, input_mask, position_ids, labels = batch
                            output = model(
                                input_ids=input_ids,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask,
                                position_ids=position_ids,
                                # cate_ids=cate_ids,
                                labels=labels,
                                output_hidden_states=True,
                                image_indices=pair_indices
                            )
                        elif args.interaction_type == "two_tower":
                            output = model(feature_matrix=feature_matrix, adjacency_matrix=edge_index, pairs=batch)
                        else:
                            raise ValueError("interaction type should be: one_tower or two_tower")
                else:
                    if args.interaction_type == "one_tower":
                        pair_indices = [t.to(device=device, non_blocking=True) for t in batch[2]]
                        batch = tuple(t.to(device=device, non_blocking=True) if t is not None else t for t in batch[3:])
                        input_ids, segment_ids, input_mask, position_ids, labels = batch
                        output = model(
                            input_ids=input_ids,
                            token_type_ids=segment_ids,
                            attention_mask=input_mask,
                            position_ids=position_ids,
                            # cate_ids=cate_ids,
                            labels=labels,
                            output_hidden_states=True,
                            image_indices=pair_indices
                        )
                    elif args.interaction_type == "two_tower":
                        output = model(feature_matrix=feature_matrix, adjacency_matrix=edge_index, pairs=batch)
                    else:
                        raise ValueError("interaction type should be: one_tower or two_tower")

                loss = output.loss
                # try:
                #     value_loss = int(loss.cpu().detach().numpy() * 1000) / 1000
                # except Exception:
                #     value_loss = loss.cpu().detach().numpy()
                if step % args.log_steps == 0:
                    logger.info(f"[Epoch-{epoch} Step-{step}] loss: {loss}")

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # 梯度回传
                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    global_step += 1
                    # 更新学习率
                    scheduler.step()

            if args.save_epochs is not None and epoch % args.save_epochs == 0:
                # Evaluation per epoch
                if args.do_eval:
                    logger.info(f'[Epoch-{epoch}] Starting evaluation ...')
                    model.eval()
                    torch.set_grad_enabled(False)

                    model_probs = None
                    model_labels = None
                    for step, batch in enumerate(valid_data_loader):
                        optimizer.zero_grad()
                        labels = np.array([int(b['item_label']) for b in batch])
                        if args.fp16:
                            with torch.cuda.amp.autocast():
                                if args.interaction_type == "one_tower":
                                    pair_indices = [t.to(device=device, non_blocking=True) for t in batch[2]]
                                    batch = tuple(t.to(device=device, non_blocking=True) if t is not None else t for t in batch[3:])
                                    input_ids, segment_ids, input_mask, position_ids, labels = batch
                                    output = model(
                                        input_ids=input_ids,
                                        token_type_ids=segment_ids,
                                        attention_mask=input_mask,
                                        position_ids=position_ids,
                                        labels=labels,
                                        output_hidden_states=True,
                                        image_indices=pair_indices
                                    )
                                elif args.interaction_type == "two_tower":
                                    output = model(feature_matrix=feature_matrix, adjacency_matrix=edge_index, pairs=batch)
                                else:
                                    raise ValueError("interaction type should be: one_tower or two_tower")
                        else:
                            if args.interaction_type == "one_tower":
                                pair_indices = [t.to(device=device, non_blocking=True) for t in batch[2]]
                                batch = tuple(t.to(device=device, non_blocking=True) if t is not None else t for t in batch[3:])
                                input_ids, segment_ids, input_mask, position_ids, labels = batch
                                output = model(
                                    input_ids=input_ids,
                                    token_type_ids=segment_ids,
                                    attention_mask=input_mask,
                                    position_ids=position_ids,
                                    labels=labels,
                                    output_hidden_states=True,
                                    image_indices=pair_indices
                                )
                            elif args.interaction_type == "two_tower":
                                output = model(feature_matrix=feature_matrix, adjacency_matrix=edge_index, pairs=batch)
                            else:
                                raise ValueError("interaction type should be: one_tower or two_tower")

                        probs = output.probs.cpu().detach().numpy()
                        # labels = labels.cpu().detach().numpy()
                        if model_probs is None:
                            model_probs = probs
                            model_labels = labels
                        else:
                            model_probs = np.append(model_probs, probs)
                            model_labels = np.append(model_labels, labels)

                    # calculate precision, recall and f1
                    for threshold in np.arange(0.1, 1.0, 0.1):
                        p = precision_score(model_labels, model_probs >= threshold)
                        r = recall_score(model_labels, model_probs >= threshold)
                        f1 = f1_score(model_labels, model_probs >= threshold)
                        logger.info(f"[Epoch-{epoch}] threshold={threshold}, precision={p}, recall={r}, f1={f1}")

                    torch.set_grad_enabled(True)

                # Model saving per epoch
                logger.info(f"[Epoch-{epoch}] saving model")
                model_to_save = (model.module if hasattr(model, "module") else model)  # Only save the model it-self
                output_model_file = os.path.join(output_model_path, f"graph_epoch-{epoch}.bin")
                torch.save(model_to_save.state_dict(), output_model_file)

    elif args.do_eval:
        model.eval()
        torch.set_grad_enabled(False)

        model_probs = None
        model_labels = None
        for step, batch in enumerate(valid_data_loader):
            labels = np.array([int(b['item_label']) for b in batch])
            if args.fp16:
                with torch.cuda.amp.autocast():
                    if args.interaction_type == "one_tower":
                        pair_indices = [t.to(device=device, non_blocking=True) for t in batch[2]]
                        batch = tuple(t.to(device=device, non_blocking=True) if t is not None else t for t in batch[3:])
                        input_ids, segment_ids, input_mask, position_ids, labels = batch
                        output = model(
                            input_ids=input_ids,
                            token_type_ids=segment_ids,
                            attention_mask=input_mask,
                            position_ids=position_ids,
                            # cate_ids=cate_ids,
                            labels=labels,
                            output_hidden_states=True,
                            image_indices=pair_indices
                        )
                    elif args.interaction_type == "two_tower":
                        output = model(feature_matrix=feature_matrix, adjacency_matrix=edge_index, pairs=batch)
                    else:
                        raise ValueError("interaction type should be: one_tower or two_tower")
            else:
                if args.interaction_type == "one_tower":
                    pair_indices = [t.to(device=device, non_blocking=True) for t in batch[2]]
                    batch = tuple(t.to(device=device, non_blocking=True) if t is not None else t for t in batch[3:])
                    input_ids, segment_ids, input_mask, position_ids, labels = batch
                    output = model(
                        input_ids=input_ids,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                        position_ids=position_ids,
                        # cate_ids=cate_ids,
                        labels=labels,
                        output_hidden_states=True,
                        image_indices=pair_indices
                    )
                elif args.interaction_type == "two_tower":
                    output = model(feature_matrix=feature_matrix, adjacency_matrix=edge_index, pairs=batch)
                else:
                    raise ValueError("interaction type should be: one_tower or two_tower")

            probs = output.probs.cpu().detach().numpy()
            # labels = labels.cpu().detach().numpy()
            if model_probs is None:
                model_probs = probs
                model_labels = labels
            else:
                model_probs = np.append(model_probs, probs)
                model_labels = np.append(model_labels, labels)

        # calculate precision, recall and f1
        for threshold in np.arange(0.1, 1.0, 0.1):
            p = precision_score(model_labels, model_probs >= threshold)
            r = recall_score(model_labels, model_probs >= threshold)
            f1 = f1_score(model_labels, model_probs >= threshold)
            logger.info(f"threshold={threshold}, precision={p}, recall={r}, f1={f1}")

    if args.do_pred:
        if "gcn" in args.model_name:
            test_dataset = GCNDataset(test_data)
            test_data_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                           collate_fn=collate_gnn)
        # elif "graph_sage" in args.model_name:
        #     if args.interaction_type == "one_tower":
        #         test_dataset = RobertaOneTowerDataset(test_data, tokenizer, max_seq_len=args.max_seq_len,
        #                                               max_seq_len_pv=args.max_seq_len_pv, classification_method=args.classification_method,
        #                                               auxiliary_task=args.auxiliary_task)
        #         test_data_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
        #                                        collate_fn=collate_one_tower)
        #     elif args.interaction_type == "two_tower":
        #         test_dataset = RobertaTwoTowerDataset(test_data, tokenizer, max_seq_en=args.max_seq_len,
        #                                                max_seq_len_pv=args.max_seq_len_pv)
        #         test_data_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
        #                                        collate_fn=collate_two_tower)
        #     else:
        #         raise ValueError("interaction type should be: one_tower or two_tower")
        else:
            raise ValueError("model name should be: roberta or pkgm")

        model.eval()
        torch.set_grad_enabled(False)

        with open(os.path.join(output_model_path, f"deepAI_result_threshold={args.threshold}.jsonl"), "w", encoding="utf-8") as w:
            for step, batch in enumerate(test_data_loader):
                src_item_ids, tgt_item_ids = [], []
                for b in batch:
                    src_item_ids.append(b['src_item_id'])
                    tgt_item_ids.append(b['tgt_item_id'])

                if args.fp16:
                    with torch.cuda.amp.autocast():
                        if args.interaction_type == "one_tower":
                            pair_indices = [t.to(device=device, non_blocking=True) for t in batch[2]]
                            batch = tuple(t.to(device=device, non_blocking=True) if t is not None else t for t in batch[3:])
                            input_ids, segment_ids, input_mask, position_ids, labels = batch
                            output = model(
                                input_ids=input_ids,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask,
                                position_ids=position_ids,
                                # cate_ids=cate_ids,
                                labels=None,
                                output_hidden_states=True,
                                image_indices=pair_indices
                            )
                        elif args.interaction_type == "two_tower":
                            output = model(feature_matrix=feature_matrix, adjacency_matrix=edge_index, pairs=batch)
                        else:
                            raise ValueError("interaction type should be: one_tower or two_tower")
                else:
                    if args.interaction_type == "one_tower":
                        pair_indices = [t.to(device=device, non_blocking=True) for t in batch[2]]
                        batch = tuple(t.to(device=device, non_blocking=True) if t is not None else t for t in batch[3:])
                        input_ids, segment_ids, input_mask, position_ids, labels = batch
                        output = model(
                            input_ids=input_ids,
                            token_type_ids=segment_ids,
                            attention_mask=input_mask,
                            position_ids=position_ids,
                            # cate_ids=cate_ids,
                            labels=None,
                            output_hidden_states=True,
                            image_indices=pair_indices
                        )
                    elif args.interaction_type == "two_tower":
                        output = model(feature_matrix=feature_matrix, adjacency_matrix=edge_index, pairs=batch)
                    else:
                        raise ValueError("interaction type should be: one_tower or two_tower")

                src_embeds = output.src_embeds.cpu().detach().numpy()
                tgt_embeds = output.tgt_embeds.cpu().detach().numpy()
                for src_item_id, tgt_item_id, src_embed, tgt_embed in zip(src_item_ids, tgt_item_ids, src_embeds, tgt_embeds):
                    src_item_emb = ','.join([str(emb) for emb in src_embed]) if isinstance(src_embed, np.ndarray) else str(src_embed)
                    tgt_item_emb = ','.join([str(emb) for emb in tgt_embed]) if isinstance(tgt_embed, np.ndarray) else str(tgt_embed)
                    rd = {"src_item_id": src_item_id, "src_item_emb": f"[{src_item_emb}]",
                          "tgt_item_id": tgt_item_id, "tgt_item_emb": f"[{tgt_item_emb}]",
                          "threshold": args.threshold}
                    w.write(json.dumps(rd) + "\n")

                if args.log_steps is not None and step % args.log_steps == 0:
                    logger.info(f"[Prediction] {step} samples processed")

        logger.info(f"[Prediction] Finished")


if __name__ == "__main__":
    main()
