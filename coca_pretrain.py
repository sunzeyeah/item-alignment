import os
import random
import argparse
import json
import jieba
import numpy as np
import torch

from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertConfig,
    get_linear_schedule_with_warmup
)
from src.models import ViT, RobertaModel, CoCaForPretraining, ROBERTA_WEIGHTS_NAME
from src.data import MultimodalDataset, collate_coca
from src.utils import logger, BOS_TOKEN
# from vit_pytorch.extractor import Extractor
# from transformers.models.roberta.modeling_roberta import RobertaModel


def get_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", required=True, type=str, help="模型训练数据地址")
    parser.add_argument("--output_dir", required=True, type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--model_name", required=True, type=str, help="model name")
    parser.add_argument("--config_file", required=True, type=str, help="model config file")
    parser.add_argument("--pretrained_text_model_path", required=True, type=str, help="pretrained text model path, RoBerta")
    parser.add_argument("--pretrained_image_model_path", required=True, type=str, help="pretrained model model path, ViT")

    # training
    parser.add_argument("--seed", default=2345, type=int, help="random seed")
    parser.add_argument("--train_batch_size", default=64, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--start_epoch", default=0, type=int, help="starting training epoch")
    parser.add_argument("--num_train_epochs", default=1000, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="weight decay")
    parser.add_argument("--log_steps", default=None, type=int, help="every n steps, log training process")
    parser.add_argument("--file_state_dict", default=None, type=str, help="finetuned model path")
    parser.add_argument("--parameters_to_freeze", default=None, type=str, help="file that contains parameters that do not require gradient descend")
    # optimization
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for. "
                                                                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit float precision instead of 32-bit")
    # cv
    parser.add_argument("--image_model_name", default="vit_base_patch16_384", type=str, help="image model name")
    parser.add_argument("--image_size", default=384, type=int, help="image height and width in pixels")
    parser.add_argument("--hflip", default=0.5, type=float, help="image transform: horizontal flip")
    parser.add_argument("--color_jitter", default=None, type=float, help="image transform: color jitter")
    # NLP
    parser.add_argument("--do_lower_case", default=True, type=bool, help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_len", default=None, type=int, help="max length for one item title")
    parser.add_argument("--max_seq_len_pv", default=None, type=int, help="max length of pvs, 'None' - do not add pvs as text")
    # Coca
    parser.add_argument("--caption_loss_weight", default=1.0, type=float, help="caption loss weight")
    parser.add_argument("--contrastive_loss_weight", default=1.0, type=float, help="constrative loss weight")

    return parser.parse_args()


def load_raw_data(args):
    train_data = []
    with open(os.path.join(args.data_dir, "item_info.jsonl"), "r", encoding="utf-8") as r:
        while True:
            line = r.readline()
            if not line:
                break
            d = json.loads(line.strip("\n"))
            title = " ".join(jieba.cut(d.get('title', "")))
            item_pvs = d.get('item_pvs', "").replace("#", "")
            sku_pvs = d.get('sku_pvs', "").replace("#", "")
            if len(item_pvs) > 0:
                if len(sku_pvs) > 0:
                    pvs = ";".join((item_pvs, sku_pvs))
                else:
                    pvs = item_pvs
            elif len(sku_pvs) > 0:
                pvs = sku_pvs
            else:
                pvs = ""
            pvs = " ".join(jieba.cut(pvs))
            item_image_name = d.get('item_image_name', "")
            image_path = os.path.join(args.data_dir, "item_images", item_image_name)
            train_data.append((d['item_id'], title, pvs, image_path))

    logger.info(f"Finished load item info, size: {len(train_data)}")

    return train_data


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
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_text_model_path, do_lower_case=args.do_lower_case)
    tokenizer.do_basic_tokenize = False
    tokenizer.bos_token = BOS_TOKEN
    logger.info(f"vocab size: {tokenizer.vocab_size}")

    # load config
    config = BertConfig.from_json_file(os.path.join(args.output_dir, args.config_file))
    config.image_size = args.image_size
    config.max_seq_len = args.max_seq_len
    config.max_seq_len_pv = args.max_seq_len_pv
    config.caption_loss_weight = args.caption_loss_weight
    config.contrastive_loss_weight = args.contrastive_loss_weight

    # load text model
    text_encoder = RobertaModel(config)
    state_dict = torch.load(os.path.join(args.pretrained_text_model_path, ROBERTA_WEIGHTS_NAME), map_location="cpu")
    text_encoder.load_state_dict(state_dict, strict=False)

    # load image model
    image_encoder = ViT(config)
    # image_encoder = timm.create_model(args.image_model_name, pretrained=True)
    # image_encoder = Extractor(image_encoder, device=device, layer_name="blocks", return_embeddings_only=True)
    image_encoder.load_pretrained(args.pretrained_image_model_path)

    # load coca pretraining model
    model = CoCaForPretraining(config, text_encoder=text_encoder, image_encoder=image_encoder)

    # 冻结部分模型参数
    if args.parameters_to_freeze is not None:
        parameters_to_freeze = json.load(open(args.parameters_to_freeze, "r", encoding="utf-8"))
        parameters_freezed = []
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
    train_data = load_raw_data(args)
    logger.info(f"# train samples: {len(train_data)}")
    train_dataset = MultimodalDataset(train_data, image_size=args.image_size, is_training=True,
                                      text_tokenizer=tokenizer, max_seq_len=args.max_seq_len,
                                      max_seq_len_pv=args.max_seq_len_pv, hflip=args.hflip,
                                      color_jitter=args.color_jitter)
    train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                   collate_fn=collate_coca)

    # optimizer
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

    if device == "cuda":
        model.cuda()
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    output_model_path = os.path.join(args.output_dir, args.model_name)
    if not os.path.exists(output_model_path):
        os.mkdir(output_model_path)

    # save all the hidden parameters.
    with open(os.path.join(output_model_path, "hyperparamter.txt"), "w") as f:
        print(args, file=f)  # Python 3.x
        print("\n", file=f)
        print(config, file=f)

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    logger.info("***** Running training *****")
    logger.info("  Model name = %s", args.model_name)
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    logger.info("  Learning rate = %.3f", args.learning_rate)

    global_step = 0
    for epoch in range(int(args.start_epoch), int(args.num_train_epochs)):
        model.train()
        for step, batch in enumerate(train_data_loader):
            batch = tuple(t.to(device=device, non_blocking=True) if t is not None else t for t in batch[1:])
            optimizer.zero_grad()
            if args.fp16:
                with torch.cuda.amp.autocast():
                    input_ids, attention_mask, token_type_ids, position_ids, images = batch
                    loss = model(input_ids, attention_mask, token_type_ids, position_ids, images)
            else:
                input_ids, attention_mask, token_type_ids, position_ids, images = batch
                loss = model(input_ids, attention_mask, token_type_ids, position_ids, images)

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

        # Model saving per epoch
        logger.info(f"[Epoch-{epoch}] saving model")
        model_to_save = (model.module if hasattr(model, "module") else model)  # Only save the model it-self
        output_model_file = os.path.join(output_model_path, f"coca_pretrain_epoch-{epoch}.bin")
        torch.save(model_to_save.state_dict(), output_model_file)


if __name__ == "__main__":
    main()
