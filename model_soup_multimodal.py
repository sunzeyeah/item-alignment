import os
import random
import argparse
import json
import numpy as np
import torch
import timm

from collections import OrderedDict
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertConfig
)
from src.models import (
    RobertaModel,
    RobertaImageOneTower,
    RobertaImageTwoTower,
    CoCaForItemAlignment,
)
from src.data import (
    RobertaImageOneTowerDataset,
    collate_multimodal,
    RobertaImageTwoTowerDataset,
    collate_multimodal_two_tower,
    PairedMultimodalDataset,
    collate_coca_pair
)
# from vit_pytorch.extractor import Extractor
# from transformers.models.roberta.modeling_roberta import RobertaModel
from src.utils import logger, ROBERTA_WEIGHTS_NAME, VIT_WEIGHTS_NAME, BOS_TOKEN


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
    parser.add_argument("--file_state_dict", required=True, type=str, help="finetuned model path")
    parser.add_argument("--epochs", required=True, type=str, help="epochs to be used for uniform soup")
    # training
    parser.add_argument("--seed", default=2345, type=int, help="random seed")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for training.")
    parser.add_argument("--start_epoch", default=0, type=int, help="starting training epoch")
    parser.add_argument("--log_steps", default=None, type=int, help="every n steps, log training process")
    parser.add_argument("--pretrained_model_path", default=None, type=str, help="pretrained model path, including roberta and pkgm")

    parser.add_argument("--type_vocab_size", default=2, type=int, help="Number of unique segment ids")
    parser.add_argument("--parameters_to_freeze", default=None, type=str, help="file that contains parameters that do not require gradient descend")
    parser.add_argument("--threshold", default=0.5, type=float, help="default threshold for item embedding score for prediction")
    parser.add_argument("--similarity_measure", default="NA", type=str,
                        help="向量相似度量: cosine, inner_product, l1 (l1 euclidean distance), l2 (l2 euclidean distance)",)
    # optimization
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
    # cv
    parser.add_argument("--image_size", default=384, type=int, help="resolution of image, height and weight")
    parser.add_argument("--image_hidden_size", default=3072, type=int, help="dimension of pretrained image embedding")
    parser.add_argument("--image_model_name", default="vit_base_patch16_384", type=str, help="image model name")
    parser.add_argument("--hflip", default=0.5, type=float, help="image transform: horizontal flip")
    parser.add_argument("--color_jitter", default=None, type=float, help="image transform: color jitter")

    return parser.parse_args()


def load_raw_data(args):
    if "roberta_image" in args.model_name:
        test_data = []
        with open(os.path.join(args.data_dir, "finetune_valid.tsv"), "r", encoding="utf-8") as r:
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

        test_data = []
        with open(os.path.join(args.data_dir, "processed", "finetune_valid.tsv"), "r", encoding="utf-8") as r:
            while True:
                line = r.readline()
                if not line:
                    break
                item_label, src_item_id, src_title, src_pvs, tgt_item_id, tgt_title, tgt_pvs = line.strip("\n").split("\t")
                src_image_path = os.path.join(args.data_dir, "raw", "item_images", id2image_name[src_item_id])
                tgt_image_path = os.path.join(args.data_dir, "raw", "item_images", id2image_name[tgt_item_id])
                test_data.append((item_label, src_item_id, src_title, src_pvs, src_image_path, tgt_item_id, tgt_title, tgt_pvs, tgt_image_path))

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
    # kg_entity_tokenizer, kg_relation_tokenizer = load_kg_tokenizer(args)
    # logger.info(f"# kg entities: {len(kg_entity_tokenizer)}, # kg relations: {len(kg_relation_tokenizer)}")
    # load model
    config = BertConfig.from_json_file(os.path.join(args.output_dir, args.config_file))
    config.interaction_type = args.interaction_type
    config.type_vocab_size = args.type_vocab_size
    config.classification_method = args.classification_method
    config.similarity_measure = args.similarity_measure
    config.loss_type = args.loss_type
    config.max_seq_len = args.max_seq_len
    config.max_seq_len_pv = args.max_seq_len_pv
    config.max_pvs = args.max_pvs
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
        if "sum" == args.ensemble:
            state_dict = torch.load(os.path.join(args.pretrained_model_path, VIT_WEIGHTS_NAME), map_location="cpu")
            image_encoder.load_state_dict(state_dict, strict=False)
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

    # Uniform model soup on previous model weights
    st = OrderedDict()
    epochs = args.epochs.split(",")
    named_parameters = dict(model.named_parameters())
    for epoch in epochs:
        state_dict = torch.load(args.file_state_dict.format(epoch), map_location="cpu")
        for key, val in state_dict.items():
            if key not in named_parameters:
                st[key] = val
            elif key not in st:
                st[key] = val
            else:
                st[key] += val
    for key, val in st.items():
        if key in named_parameters:
            val /= len(epochs)
    model.load_state_dict(st)
    logger.info(f"Finished uniform soup on epochs: {epochs}")

    # save uniform soup state dict
    model_path = f"{args.model_name}-{args.data_version}-{args.interaction_type}-{args.classification_method}-{args.ensemble}-{args.loss_type}"
    output_model_path = os.path.join(args.output_dir, model_path)
    if not os.path.exists(output_model_path):
        os.mkdir(output_model_path)
    output_model_file = os.path.join(output_model_path, f"multimodal_finetune-uniform_soup-epoch-{args.epochs}.bin")
    torch.save(st, output_model_file)
    logger.info(f"Finished saving uniform soup model")

    # load raw data
    test_data = load_raw_data(args)
    logger.info(f"# test samples: {len(test_data)}")

    if device == "cuda":
        model.cuda()

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

    with open(os.path.join(output_model_path, f"deepAI_result_uniform_soup_threshold={args.threshold}.jsonl"), "w", encoding="utf-8") as w:
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

    logger.info(f"[Prediction] Finished processing")


if __name__ == "__main__":
    main()
