import os
import random
import argparse
import json
import numpy as np
import torch
import timm

from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertConfig,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import precision_score, recall_score, f1_score
from src.models import (
    RobertaModel,
    RobertaImageOneTower,
    RobertaImageTwoTower,
    CoCaForItemAlignment
)
from src.data import (
    RobertaImageOneTowerDataset,
    RobertaImageTwoTowerDataset,
    collate_multimodal,
    PairedMultimodalDataset,
    collate_coca_pair,
    collate_multimodal_two_tower
)
from src.utils import logger, VIT_WEIGHTS_NAME, ROBERTA_WEIGHTS_NAME, BOS_TOKEN


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
    parser.add_argument("--ensemble", required=True, type=str, help="ways to ensemble different modalities, values: begin, end, sum, cross_attn"
                                                                    "begin - (roberta image) concat image embeds with word embeds as input embeddings,"
                                                                    "end - (roberta image) concat image embeds with [CLS] hidden states,"
                                                                    "sum - (coca) sum text embeds and image embeds as final embeds,"
                                                                    "cross_attn - (coca) cross attention between text and image sequence output, use postition at 0 as final embeds")
    parser.add_argument("--loss_type", required=True, type=str,
                        help="损失函数类型, ce: cross entropy, bce: binary cross entropy with logits loss, cosine: cosine embedding loss",)
    # training
    parser.add_argument("--do_train", action="store_true", help="是否进行模型训练")
    parser.add_argument("--do_eval", action="store_true", help="是否进行模型验证")
    parser.add_argument("--do_pred", action="store_true", help="是否进行模型测试")
    parser.add_argument("--seed", default=2345, type=int, help="random seed")
    parser.add_argument("--train_batch_size", default=64, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--start_epoch", default=0, type=int, help="starting training epoch")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="weight decay")
    parser.add_argument("--log_steps", default=None, type=int, help="every n steps, log training process")
    parser.add_argument("--pretrained_model_path", default=None, type=str, help="pretrained model path, including roberta and pkgm")
    parser.add_argument("--file_state_dict", default=None, type=str, help="finetuned model path")
    parser.add_argument("--type_vocab_size", default=2, type=int, help="Number of unique segment ids")
    parser.add_argument("--parameters_to_freeze", default=None, type=str, help="file that contains parameters that do not require gradient descend")
    parser.add_argument("--threshold", default=0.5, type=float, help="default threshold for item embedding score for prediction")
    parser.add_argument("--similarity_measure", default="NA", type=str,
                        help="向量相似度量: cosine, inner_product, l1 (l1 euclidean distance), l2 (l2 euclidean distance)",)
    # optimization
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for. "
                                                                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--margin", default=1.0, type=float, help="margin in loss function")
    # NLP
    parser.add_argument("--do_lower_case", default=True, type=bool, help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_len", default=50, type=int, help="max length for one item title")
    parser.add_argument("--max_seq_len_pv", default=305, type=int, help="max length of pvs, 'None' - do not add pvs as text")
    parser.add_argument("--max_position_embeddings", default=512, type=int, help="max position embedding length")
    parser.add_argument("--cls_layers", default="1", type=str, help="which layers of cls used for classification")
    parser.add_argument("--cls_pool", default="cat", type=str, help="ways to pool multiple layers of cls used for classification")
    # cv
    parser.add_argument("--image_size", default=384, type=int, help="resolution of image, height and weight")
    parser.add_argument("--image_hidden_size", default=3072, type=int, help="dimension of pretrained image embedding")
    parser.add_argument("--image_model_name", default="vit_base_patch16_384", type=str, help="image model name")
    parser.add_argument("--hflip", default=0.5, type=float, help="image transform: horizontal flip")
    parser.add_argument("--color_jitter", default=None, type=float, help="image transform: color jitter")

    return parser.parse_args()


def load_raw_data(args):
    if "roberta_image" in args.model_name:
        train_data = []
        with open(os.path.join(args.data_dir, "finetune_train.tsv"), "r", encoding="utf-8") as r:
            while True:
                line = r.readline()
                if not line:
                    break
                train_data.append(line.strip("\n").split("\t"))

        valid_data = []
        with open(os.path.join(args.data_dir, "finetune_test.tsv"), "r", encoding="utf-8") as r:
            while True:
                line = r.readline()
                if not line:
                    break
                valid_data.append(line.strip("\n").split("\t"))

        test_data = []
        with open(os.path.join(args.data_dir, "finetune_test.tsv"), "r", encoding="utf-8") as r:
            while True:
                line = r.readline()
                if not line:
                    break
                test_data.append(line.strip("\n").split("\t"))
    else:
        id2image_name = dict()
        with open(os.path.join(args.data_dir, "raw", "item_info.jsonl"), "r", encoding="utf-8") as r:
            while True:
                line = r.readline()
                if not line:
                    break
                d = json.loads(line.strip())
                id2image_name[d['item_id']] = d['item_image_name']
        logger.info(f"Finished load item info, size: {len(id2image_name)}")

        train_data = []
        with open(os.path.join(args.data_dir, "processed", "finetune_train.tsv"), "r", encoding="utf-8") as r:
            while True:
                line = r.readline()
                if not line:
                    break
                item_label, src_item_id, src_title, src_pvs, tgt_item_id, tgt_title, tgt_pvs = line.strip("\n").split("\t")
                src_image_path = os.path.join(args.data_dir, "raw", "item_images", id2image_name[src_item_id])
                tgt_image_path = os.path.join(args.data_dir, "raw", "item_images", id2image_name[tgt_item_id])
                train_data.append((item_label, src_item_id, src_title, src_pvs, src_image_path, tgt_item_id, tgt_title, tgt_pvs, tgt_image_path))

        valid_data = []
        with open(os.path.join(args.data_dir, "processed", "finetune_test.tsv"), "r", encoding="utf-8") as r:
            while True:
                line = r.readline()
                if not line:
                    break
                item_label, src_item_id, src_title, src_pvs, tgt_item_id, tgt_title, tgt_pvs = line.strip("\n").split("\t")
                src_image_path = os.path.join(args.data_dir, "raw", "item_images", id2image_name[src_item_id])
                tgt_image_path = os.path.join(args.data_dir, "raw", "item_images", id2image_name[tgt_item_id])
                valid_data.append((item_label, src_item_id, src_title, src_pvs, src_image_path, tgt_item_id, tgt_title, tgt_pvs, tgt_image_path))

        test_data = []
        with open(os.path.join(args.data_dir, "processed", "finetune_test.tsv"), "r", encoding="utf-8") as r:
            while True:
                line = r.readline()
                if not line:
                    break
                item_label, src_item_id, src_title, src_pvs, tgt_item_id, tgt_title, tgt_pvs = line.strip("\n").split("\t")
                src_image_path = os.path.join(args.data_dir, "raw", "item_images", id2image_name[src_item_id])
                tgt_image_path = os.path.join(args.data_dir, "raw", "item_images", id2image_name[tgt_item_id])
                test_data.append((item_label, src_item_id, src_title, src_pvs, src_image_path, tgt_item_id, tgt_title, tgt_pvs, tgt_image_path))

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
    config.type_vocab_size = args.type_vocab_size
    config.classification_method = args.classification_method
    config.similarity_measure = args.similarity_measure
    config.loss_type = args.loss_type
    config.max_seq_len = args.max_seq_len
    config.max_seq_len_pv = args.max_seq_len_pv
    config.max_position_embeddings = args.max_position_embeddings
    config.loss_margin = args.margin
    config.cls_layers = args.cls_layers
    config.cls_pool = args.cls_pool
    config.ensemble = args.ensemble
    config.image_hidden_size = args.image_hidden_size
    config.image_size = args.image_size
    if args.max_seq_len_pv is None:
        max_seq_len = args.max_seq_len
    elif args.max_seq_len is None:
        max_seq_len = args.max_seq_len_pv
    else:
        max_seq_len = args.max_seq_len + args.max_seq_len_pv
    if args.interaction_type == "one_tower":
        assert args.max_position_embeddings >= 2 * max_seq_len + 2

    if "roberta_image" in args.model_name:
        if args.interaction_type == "one_tower":
            model = RobertaImageOneTower.from_pretrained(args.pretrained_model_path, config=config,
                                                         ignore_mismatched_sizes=True)
        elif args.interaction_type == "two_tower":
            model = RobertaImageTwoTower.from_pretrained(args.pretrained_model_path, config=config,
                                                         ignore_mismatched_sizes=True)
        else:
            raise ValueError("interaction type should be: one_tower or two_tower")
    elif "coca" in args.model_name:
        # load text model
        text_encoder = RobertaModel.from_pretrained(args.pretrained_model_path, config=config)
        # load image model
        # image_encoder = ViT(config)
        image_encoder = timm.create_model(args.image_model_name, pretrained=True)
        # image_encoder = Extractor(image_encoder, device=device, layer_name="blocks", return_embeddings_only=True)
        # if "sum" == args.ensemble:
        #     state_dict = torch.load(os.path.join(args.pretrained_model_path, VIT_WEIGHTS_NAME), map_location="cpu")
        #     image_encoder.load_state_dict(state_dict, strict=False)
        # load coca model
        model = CoCaForItemAlignment(config, image_encoder, text_encoder)
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    # 如果type token vocab大于2，则将原始embedding矩阵的前2行复制到模型weight
    if args.type_vocab_size > 2:
        state_dict_pretrained = torch.load(os.path.join(args.pretrained_model_path, ROBERTA_WEIGHTS_NAME), map_location="cpu")
        token_type_embedding = state_dict_pretrained['embeddings.token_type_embeddings.weight']
        token_type_embedding_orig = model.state_dict()['roberta.embeddings.token_type_embeddings.weight']
        token_type_embedding_orig[:2, :] = token_type_embedding
        model.state_dict()['roberta.embeddings.token_type_embeddings.weight'] = token_type_embedding_orig
    # 如果max_position_embeddings大于512，则将原始embedding矩阵的前512行复制到模型weight
    if args.max_position_embeddings > 512:
        state_dict_pretrained = torch.load(os.path.join(args.pretrained_model_path, ROBERTA_WEIGHTS_NAME), map_location="cpu")
        position_embedding = state_dict_pretrained['embeddings.position_embeddings.weight']
        position_embedding_orig = model.state_dict()['roberta.embeddings.position_embeddings.weight']
        position_embedding_orig[:512, :] = position_embedding
        model.state_dict()['roberta.embeddings.position_embeddings.weight'] = position_embedding_orig
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

    if args.do_train:
        if "roberta_image" in args.model_name:
            if args.interaction_type == "one_tower":
                train_dataset = RobertaImageOneTowerDataset(train_data, tokenizer, max_seq_len=args.max_seq_len,
                                                            max_seq_len_pv=args.max_seq_len_pv, ensemble=args.ensemble)
                train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                               collate_fn=collate_multimodal)
            elif args.interaction_type == "two_tower":
                train_dataset = RobertaImageTwoTowerDataset(train_data, tokenizer, max_seq_len=args.max_seq_len,
                                                            max_seq_len_pv=args.max_seq_len_pv, ensemble=args.ensemble)
                train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                               collate_fn=collate_multimodal_two_tower)
            else:
                raise ValueError("interaction type should be: one_tower or two_tower")
        elif "coca" in args.model_name:
            train_dataset = PairedMultimodalDataset(train_data, ensemble=args.ensemble, image_size=args.image_size,
                                                    is_training=True, text_tokenizer=tokenizer,
                                                    max_seq_len=args.max_seq_len, max_seq_len_pv=args.max_seq_len_pv,
                                                    hflip=args.hflip, color_jitter=args.color_jitter)
            train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                           collate_fn=collate_coca_pair)
        else:
            raise ValueError(f"Unsupported model name: {args.model_name}")

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
        if "roberta_image" in args.model_name:
            if args.interaction_type == "one_tower":
                valid_dataset = RobertaImageOneTowerDataset(valid_data, tokenizer, max_seq_len=args.max_seq_len,
                                                            max_seq_len_pv=args.max_seq_len_pv, ensemble=args.ensemble)
                valid_data_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                               collate_fn=collate_multimodal)
            elif args.interaction_type == "two_tower":
                valid_dataset = RobertaImageTwoTowerDataset(valid_data, tokenizer, max_seq_len=args.max_seq_len,
                                                            max_seq_len_pv=args.max_seq_len_pv, ensemble=args.ensemble)
                valid_data_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                               collate_fn=collate_multimodal_two_tower)
            else:
                raise ValueError("interaction type should be: one_tower or two_tower")
        elif "coca" in args.model_name:
            valid_dataset = PairedMultimodalDataset(train_data, ensemble=args.ensemble,image_size=args.image_size,
                                                    is_training=False, text_tokenizer=tokenizer,
                                                    max_seq_len=args.max_seq_len, max_seq_len_pv=args.max_seq_len_pv,
                                                    hflip=args.hflip, color_jitter=args.color_jitter)
            valid_data_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                           collate_fn=collate_coca_pair)
        else:
            raise ValueError(f"Unsupported model name: {args.model_name}")

    if device == "cuda":
        model.cuda()
        if args.do_train:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

    model_path = f"{args.model_name}-{args.data_version}-{args.interaction_type}-{args.classification_method}-{args.ensemble}-{args.loss_type}"
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
        logger.info("  Learning rate = %.3f", args.learning_rate)

        global_step = 0
        for epoch in range(int(args.start_epoch), int(args.num_train_epochs)):
            model.train()
            for step, batch in enumerate(train_data_loader):
                batch = tuple(t.to(device=device, non_blocking=True) if t is not None else t for t in batch[2:])
                optimizer.zero_grad()
                if args.fp16:
                    with torch.cuda.amp.autocast():
                        if args.interaction_type == "one_tower":
                            image_indices, src_image_embs, tgt_image_embs, input_ids, segment_ids, input_mask, position_ids, labels = batch
                            inputs_embeds = [src_image_embs, tgt_image_embs]
                            output = model(
                                input_ids=input_ids,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask,
                                position_ids=position_ids,
                                labels=labels,
                                output_hidden_states=True,
                                inputs_embeds=inputs_embeds,
                                image_indices=image_indices
                            )
                        elif args.interaction_type == "two_tower":
                            input_ids_1, attention_mask_1, token_type_ids_1, position_ids_1, images_1, \
                                input_ids_2, attention_mask_2, token_type_ids_2, position_ids_2, images_2, labels = batch
                            output = model(
                                input_ids_1=input_ids_1,
                                token_type_ids_1=token_type_ids_1,
                                attention_mask_1=attention_mask_1,
                                position_ids_1=position_ids_1,
                                images_1=images_1,
                                input_ids_2=input_ids_2,
                                token_type_ids_2=token_type_ids_2,
                                attention_mask_2=attention_mask_2,
                                position_ids_2=position_ids_2,
                                images_2=images_2,
                                labels=labels
                            )
                        else:
                            raise ValueError("interaction type should be: one_tower or two_tower")
                else:
                    if args.interaction_type == "one_tower":
                        image_indices, src_image_embs, tgt_image_embs, input_ids, segment_ids, input_mask, position_ids, labels = batch
                        inputs_embeds = [src_image_embs, tgt_image_embs]
                        output = model(
                            input_ids=input_ids,
                            token_type_ids=segment_ids,
                            attention_mask=input_mask,
                            position_ids=position_ids,
                            labels=labels,
                            output_hidden_states=True,
                            inputs_embeds=inputs_embeds,
                            image_indices=image_indices
                        )
                    elif args.interaction_type == "two_tower":
                        input_ids_1, attention_mask_1, token_type_ids_1, position_ids_1, images_1, \
                            input_ids_2, attention_mask_2, token_type_ids_2, position_ids_2, images_2, labels = batch
                        output = model(
                            input_ids_1=input_ids_1,
                            token_type_ids_1=token_type_ids_1,
                            attention_mask_1=attention_mask_1,
                            position_ids_1=position_ids_1,
                            images_1=images_1,
                            input_ids_2=input_ids_2,
                            token_type_ids_2=token_type_ids_2,
                            attention_mask_2=attention_mask_2,
                            position_ids_2=position_ids_2,
                            images_2=images_2,
                            labels=labels
                        )
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

            # Evaluation per epoch
            if args.do_eval:
                logger.info(f'[Epoch-{epoch}] Starting evaluation ...')
                model.eval()
                torch.set_grad_enabled(False)

                model_probs = None
                model_labels = None
                for step, batch in enumerate(valid_data_loader):
                    batch = tuple(t.to(device=device, non_blocking=True) if t is not None else t for t in batch[2:])
                    optimizer.zero_grad()
                    if args.fp16:
                        with torch.cuda.amp.autocast():
                            if args.interaction_type == "one_tower":
                                image_indices, src_image_embs, tgt_image_embs, input_ids, segment_ids, input_mask, position_ids, labels = batch
                                inputs_embeds = [src_image_embs, tgt_image_embs]
                                output = model(
                                    input_ids=input_ids,
                                    token_type_ids=segment_ids,
                                    attention_mask=input_mask,
                                    position_ids=position_ids,
                                    labels=labels,
                                    output_hidden_states=True,
                                    inputs_embeds=inputs_embeds,
                                    image_indices=image_indices
                                )
                            elif args.interaction_type == "two_tower":
                                input_ids_1, attention_mask_1, token_type_ids_1, position_ids_1, images_1, \
                                    input_ids_2, attention_mask_2, token_type_ids_2, position_ids_2, images_2, labels = batch
                                output = model(
                                    input_ids_1=input_ids_1,
                                    token_type_ids_1=token_type_ids_1,
                                    attention_mask_1=attention_mask_1,
                                    position_ids_1=position_ids_1,
                                    images_1=images_1,
                                    input_ids_2=input_ids_2,
                                    token_type_ids_2=token_type_ids_2,
                                    attention_mask_2=attention_mask_2,
                                    position_ids_2=position_ids_2,
                                    images_2=images_2,
                                    labels=labels
                                )
                            else:
                                raise ValueError("interaction type should be: one_tower or two_tower")
                    else:
                        if args.interaction_type == "one_tower":
                            image_indices, src_image_embs, tgt_image_embs, input_ids, segment_ids, input_mask, position_ids, labels = batch
                            inputs_embeds = [src_image_embs, tgt_image_embs]
                            output = model(
                                input_ids=input_ids,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask,
                                position_ids=position_ids,
                                labels=labels,
                                output_hidden_states=True,
                                inputs_embeds=inputs_embeds,
                                image_indices=image_indices
                            )
                        elif args.interaction_type == "two_tower":
                            input_ids_1, attention_mask_1, token_type_ids_1, position_ids_1, images_1, \
                                input_ids_2, attention_mask_2, token_type_ids_2, position_ids_2, images_2, labels = batch
                            output = model(
                                input_ids_1=input_ids_1,
                                token_type_ids_1=token_type_ids_1,
                                attention_mask_1=attention_mask_1,
                                position_ids_1=position_ids_1,
                                images_1=images_1,
                                input_ids_2=input_ids_2,
                                token_type_ids_2=token_type_ids_2,
                                attention_mask_2=attention_mask_2,
                                position_ids_2=position_ids_2,
                                images_2=images_2,
                                labels=labels
                            )
                        else:
                            raise ValueError("interaction type should be: one_tower or two_tower")

                    probs = output.probs.cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()
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
            output_model_file = os.path.join(output_model_path, f"multimodal_finetune_epoch-{epoch}.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

    elif args.do_eval:
        model.eval()
        torch.set_grad_enabled(False)

        model_probs = None
        model_labels = None
        for step, batch in enumerate(valid_data_loader):
            batch = tuple(t.to(device=device, non_blocking=True) if t is not None else t for t in batch[2:])
            if args.fp16:
                with torch.cuda.amp.autocast():
                    if args.interaction_type == "one_tower":
                        image_indices, src_image_embs, tgt_image_embs, input_ids, segment_ids, input_mask, position_ids, labels = batch
                        inputs_embeds = [src_image_embs, tgt_image_embs]
                        output = model(
                            input_ids=input_ids,
                            token_type_ids=segment_ids,
                            attention_mask=input_mask,
                            position_ids=position_ids,
                            labels=labels,
                            output_hidden_states=True,
                            inputs_embeds=inputs_embeds,
                            image_indices=image_indices
                        )
                    elif args.interaction_type == "two_tower":
                        input_ids_1, attention_mask_1, token_type_ids_1, position_ids_1, images_1, \
                            input_ids_2, attention_mask_2, token_type_ids_2, position_ids_2, images_2, labels = batch
                        output = model(
                            input_ids_1=input_ids_1,
                            token_type_ids_1=token_type_ids_1,
                            attention_mask_1=attention_mask_1,
                            position_ids_1=position_ids_1,
                            images_1=images_1,
                            input_ids_2=input_ids_2,
                            token_type_ids_2=token_type_ids_2,
                            attention_mask_2=attention_mask_2,
                            position_ids_2=position_ids_2,
                            images_2=images_2,
                            labels=labels
                        )
                    else:
                        raise ValueError("interaction type should be: one_tower or two_tower")
            else:
                if args.interaction_type == "one_tower":
                    image_indices, src_image_embs, tgt_image_embs, input_ids, segment_ids, input_mask, position_ids, labels = batch
                    inputs_embeds = [src_image_embs, tgt_image_embs]
                    output = model(
                        input_ids=input_ids,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                        position_ids=position_ids,
                        labels=labels,
                        output_hidden_states=True,
                        inputs_embeds=inputs_embeds,
                        image_indices=image_indices
                    )
                elif args.interaction_type == "two_tower":
                    input_ids_1, attention_mask_1, token_type_ids_1, position_ids_1, images_1, \
                        input_ids_2, attention_mask_2, token_type_ids_2, position_ids_2, images_2, labels = batch
                    output = model(
                        input_ids_1=input_ids_1,
                        token_type_ids_1=token_type_ids_1,
                        attention_mask_1=attention_mask_1,
                        position_ids_1=position_ids_1,
                        images_1=images_1,
                        input_ids_2=input_ids_2,
                        token_type_ids_2=token_type_ids_2,
                        attention_mask_2=attention_mask_2,
                        position_ids_2=position_ids_2,
                        images_2=images_2,
                        labels=labels
                    )
                else:
                    raise ValueError("interaction type should be: one_tower or two_tower")

            probs = output.probs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
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
        if "roberta_image" in args.model_name:
            if args.interaction_type == "one_tower":
                test_dataset = RobertaImageOneTowerDataset(test_data, tokenizer, max_seq_len=args.max_seq_len,
                                                            max_seq_len_pv=args.max_seq_len_pv, ensemble=args.ensemble)
                test_data_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                               collate_fn=collate_multimodal)
            elif args.interaction_type == "two_tower":
                test_dataset = RobertaImageTwoTowerDataset(test_data, tokenizer, max_seq_len=args.max_seq_len,
                                                            max_seq_len_pv=args.max_seq_len_pv, ensemble=args.ensemble)
                test_data_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                               collate_fn=collate_multimodal_two_tower)
            else:
                raise ValueError("interaction type should be: one_tower or two_tower")
        elif "coca" in args.model_name:
            test_dataset = PairedMultimodalDataset(test_data, ensemble=args.ensemble, image_size=args.image_size,
                                                   is_training=False, text_tokenizer=tokenizer,
                                                   max_seq_len=args.max_seq_len, max_seq_len_pv=args.max_seq_len_pv,
                                                   hflip=args.hflip, color_jitter=args.color_jitter)
            test_data_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                          collate_fn=collate_coca_pair)
        else:
            raise ValueError(f"Unsupported model name: {args.model_name}")

        model.eval()
        torch.set_grad_enabled(False)

        out_proj = {"w": model.classifier.out_proj.weight.cpu().detach().numpy().tolist(), "b": model.classifier.out_proj.bias.cpu().detach().numpy().tolist()}
        fw = os.path.join(output_model_path, f"weights.json")
        json.dump(out_proj, open(fw, "w", encoding="utf-8"), ensure_ascii=False)

        with open(os.path.join(output_model_path, f"deepAI_result_threshold={args.threshold}.jsonl"), "w", encoding="utf-8") as w:
            for step, batch in enumerate(test_data_loader):
                src_item_ids, tgt_item_ids = batch[:2]
                batch = tuple(t.to(device=device, non_blocking=True) if t is not None else t for t in batch[2:])
                if args.fp16:
                    with torch.cuda.amp.autocast():
                        if args.interaction_type == "one_tower":
                            image_indices, src_image_embs, tgt_image_embs, input_ids, segment_ids, input_mask, position_ids, labels = batch
                            inputs_embeds = [src_image_embs, tgt_image_embs]
                            output = model(
                                input_ids=input_ids,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask,
                                position_ids=position_ids,
                                labels=None,
                                output_hidden_states=True,
                                inputs_embeds=inputs_embeds,
                                image_indices=image_indices
                            )
                        elif args.interaction_type == "two_tower":
                            input_ids_1, attention_mask_1, token_type_ids_1, position_ids_1, images_1, \
                                input_ids_2, attention_mask_2, token_type_ids_2, position_ids_2, images_2, labels = batch
                            output = model(
                                input_ids_1=input_ids_1,
                                token_type_ids_1=token_type_ids_1,
                                attention_mask_1=attention_mask_1,
                                position_ids_1=position_ids_1,
                                images_1=images_1,
                                input_ids_2=input_ids_2,
                                token_type_ids_2=token_type_ids_2,
                                attention_mask_2=attention_mask_2,
                                position_ids_2=position_ids_2,
                                images_2=images_2,
                                labels=labels
                            )
                        else:
                            raise ValueError("interaction type should be: one_tower or two_tower")
                else:
                    if args.interaction_type == "one_tower":
                        image_indices, src_image_embs, tgt_image_embs, input_ids, segment_ids, input_mask, position_ids, labels = batch
                        inputs_embeds = [src_image_embs, tgt_image_embs]
                        output = model(
                            input_ids=input_ids,
                            token_type_ids=segment_ids,
                            attention_mask=input_mask,
                            position_ids=position_ids,
                            labels=None,
                            output_hidden_states=True,
                            inputs_embeds=inputs_embeds,
                            image_indices=image_indices
                        )
                    elif args.interaction_type == "two_tower":
                        input_ids_1, attention_mask_1, token_type_ids_1, position_ids_1, images_1, \
                            input_ids_2, attention_mask_2, token_type_ids_2, position_ids_2, images_2, labels = batch
                        output = model(
                            input_ids_1=input_ids_1,
                            token_type_ids_1=token_type_ids_1,
                            attention_mask_1=attention_mask_1,
                            position_ids_1=position_ids_1,
                            images_1=images_1,
                            input_ids_2=input_ids_2,
                            token_type_ids_2=token_type_ids_2,
                            attention_mask_2=attention_mask_2,
                            position_ids_2=position_ids_2,
                            images_2=images_2,
                            labels=labels
                        )
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
