import os
import random
import argparse
import json
import numpy as np
import torch
import jieba

from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertConfig
)
from src.models import RobertaModel
from src.data import RobertaDataset, collate
from src.utils import logger, BOS_TOKEN


def get_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", required=True, type=str, help="模型训练数据地址")
    parser.add_argument("--output_dir", required=True, type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--config_file", required=True, type=str, help="The config file which specified the model details.")

    # training
    parser.add_argument("--seed", default=2345, type=int, help="random seed")
    parser.add_argument("--train_batch_size", default=64, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--start_epoch", default=0, type=int, help="starting training epoch")
    parser.add_argument("--num_train_epochs", default=1000, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="weight decay")
    parser.add_argument("--log_steps", default=None, type=int, help="every n steps, log training process")
    parser.add_argument("--pretrained_model_path", default=None, type=str, help="pretrained model path, including roberta and pkgm")
    parser.add_argument("--file_state_dict", default=None, type=str, help="finetuned model path")
    parser.add_argument("--type_vocab_size", default=2, type=int, help="Number of unique segment ids")
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
    parser.add_argument("--max_seq_len", default=None, type=int, help="max length for one item title")
    parser.add_argument("--max_seq_len_pv", default=None, type=int, help="max length of pvs, 'None' - do not add pvs as text")
    parser.add_argument("--max_position_embeddings", default=512, type=int, help="max position embedding length")
    parser.add_argument("--max_pvs", default=20, type=int, help="max number of pairs for one item")
    parser.add_argument("--cls_layers", default="1", type=str, help="which layers of cls used for classification")
    parser.add_argument("--cls_pool", default="cat", type=str, help="ways to pool multiple layers of cls used for classification")
    parser.add_argument("--auxiliary_task", action="store_true", help="whether to include auxiliary task. The task is additionally comparing pv pairs of src and tgt item."
                                                                      "for pv keys that are shared by two items, compute whether the pv value is the same")
    # TextCNN
    parser.add_argument("--filter_sizes", default="1,2,3,5", type=str, help="filter sizes")
    parser.add_argument("--num_filters", default=36, type=int, help="number of filters")

    return parser.parse_args()


def load_raw_data(args):
    id2image_name = dict()
    with open(os.path.join(args.data_dir, "raw", "item_info.jsonl"), "r", encoding="utf-8") as r:
        while True:
            line = r.readline()
            if not line:
                break
            d = json.loads(line.strip())
            id2image_name[d['item_id']] = d#['item_image_name']
    logger.info(f"Finished loading item info, size: {len(id2image_name)}")

    test_data = []
    with open(os.path.join(args.data_dir, "processed", "entity2id.txt"), "r", encoding="utf-8") as r:
        while True:
            line = r.readline()
            if not line:
                break
            item, idx = line.strip("\n").split("\t")
            if "/item/" in item:
                item = item.replace("/item/", "")
                text = id2image_name[item]['title']
            elif "/value/" in item:
                text = item.replace("/value/", "")
            else:
                logger.warning(f"wrong format data: {item}")
            text = " ".join(jieba.cut(text))
            test_data.append((idx, text))

    return test_data


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
    # config.interaction_type = args.interaction_type
    # config.type_vocab_size = args.type_vocab_size
    # config.classification_method = args.classification_method
    # config.similarity_measure = args.similarity_measure
    # config.loss_type = args.loss_type
    config.max_seq_len = args.max_seq_len
    # config.max_seq_len_pv = args.max_seq_len_pv
    # config.max_pvs = args.max_pvs
    # config.max_position_embeddings = args.max_position_embeddings
    # config.loss_margin = args.margin
    # config.cls_layers = args.cls_layers
    # config.cls_pool = args.cls_pool
    # config.filter_sizes = args.filter_sizes
    # config.num_filters = args.num_filters
    # config.auxiliary_task = args.auxiliary_task
    config.ensemble = None
    if args.max_seq_len_pv is None:
        max_seq_len = args.max_seq_len
    elif args.max_seq_len is None:
        max_seq_len = args.max_seq_len_pv
    else:
        max_seq_len = args.max_seq_len + args.max_seq_len_pv
    assert args.max_position_embeddings >= 2 * max_seq_len + 2

    model = RobertaModel.from_pretrained(args.pretrained_model_path, config=config,
                                         ignore_mismatched_sizes=True)
    # load previous model weights (if exists)
    if args.file_state_dict is not None:
        state_dict = torch.load(args.file_state_dict, map_location="cpu")
        model.load_state_dict(state_dict)
    # load raw data
    test_data = load_raw_data(args)
    logger.info(f"# test samples: {len(test_data)}")

    if device == "cuda":
        model.cuda()

    test_dataset = RobertaDataset(test_data, tokenizer,  max_seq_len=args.max_seq_len)
    test_data_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                   collate_fn=collate)

    model.eval()
    torch.set_grad_enabled(False)

    f = os.path.join(args.data_dir, "processed", f"feature_matrix.pt")
    feature_matrix = None
    for step, batch in enumerate(test_data_loader):
        ids = batch[0]
        if args.fp16:
            with torch.cuda.amp.autocast():
                batch = tuple(t.to(device=device, non_blocking=True) if t is not None else t for t in batch[1:])
                input_ids, segment_ids, input_mask, position_ids = batch
                output = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    position_ids=position_ids,
                    output_hidden_states=True
                )
        else:
            batch = tuple(t.to(device=device, non_blocking=True) if t is not None else t for t in batch[1:])
            input_ids, segment_ids, input_mask, position_ids = batch
            output = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                position_ids=position_ids,
                output_hidden_states=True
            )
        sequence_outputs = output.pooler_output
        if feature_matrix is None:
            feature_matrix = sequence_outputs
        else:
            feature_matrix = torch.cat((feature_matrix, sequence_outputs), dim=0)

        if args.log_steps is not None and step % args.log_steps == 0:
            logger.info(f"[Prediction] {step} samples processed")

    torch.save(feature_matrix, f)

    logger.info(f"[Prediction] Finished processing")


if __name__ == "__main__":
    main()
