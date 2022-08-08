import os
import random
import argparse
import json
import numpy as np
import torch
import timm

from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import BertConfig, get_linear_schedule_with_warmup
from src.models import NFNetTwoTower, VitTwoTower, ResNetTwoTower
from src.data import PairedImageDataset, collate_image
from src.utils import logger


def get_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", required=True, type=str, help="模型训练数据地址")
    parser.add_argument("--output_dir", required=True, type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--model_name", required=True, type=str, help="model name: resnetv2_50, resnetv2_50x3_bitm_in21k"
                                                                      "eca_nfnet_l0, eca_nfnet_l1, eca_nfnet_l2"
                                                                      "vit_base_patch16_384, vit_large_patch16_384"
                                                                      "vit_base_r50_s16_384, vit_base_r50_s16_224_in21k")
    parser.add_argument("--data_version", required=True, type=str, help="data version")

    # training
    parser.add_argument("--do_train", action="store_true", help="是否进行模型训练")
    parser.add_argument("--do_eval", action="store_true", help="是否进行模型验证")
    parser.add_argument("--do_pred", action="store_true", help="是否进行模型测试")
    parser.add_argument("--seed", default=2345, type=int, help="random seed")
    parser.add_argument("--config_file", default=None, type=str, help="The config file which specified the model details.")
    parser.add_argument("--train_batch_size", default=64, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--start_epoch", default=0, type=int, help="starting training epoch")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="weight decay")
    parser.add_argument("--log_steps", default=10, type=int, help="every n steps, log training process")
    parser.add_argument("--pretrained_model_path", default=None, type=str, help="pretrained model path, including roberta and pkgm")
    parser.add_argument("--file_state_dict", default=None, type=str, help="finetuned model path")
    parser.add_argument("--parameters_to_freeze", default=None, type=str, help="file that contains parameters that do not require gradient descend")
    parser.add_argument("--threshold", default=0.5, type=float, help="default threshold for item embedding score for prediction")
    # optimization
    parser.add_argument("--warmup_proportion", default=0.3, type=float, help="Proportion of training to perform linear learning rate warmup for. "
                                                                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--margin", default=1.0, type=float, help="margin in loss function")
    parser.add_argument("--loss_type", default="ce", type=str,
                        help="损失函数类型, ce: cross entropy, bce: binary cross entropy with logits loss, cosine: cosine embedding loss",)
    # cv
    parser.add_argument("--image_size", default=1000, type=int, help="image height and width in pixels")
    parser.add_argument("--hflip", default=0.5, type=float, help="image transform: horizontal flip")
    parser.add_argument("--color_jitter", default=None, type=float, help="image transform: color jitter")
    parser.add_argument("--num_classes", default=2, type=int, help="number of label classes")
    parser.add_argument("--in_chans", default=3, type=int, help="number of in channels")
    parser.add_argument("--global_pool", default="avg", type=str, help="global pooling method")
    parser.add_argument("--stride", default=32, type=int, help="stride")
    parser.add_argument("--depths", default="2,4,12,6", type=str, help="depths")
    parser.add_argument("--channels", default="256,512,1536,1536", type=str, help="channels")
    parser.add_argument("--stem_type", default="deep_quad", type=str, help="stem type")
    parser.add_argument("--stem_chs", default=128, type=int, help="stem chs")
    parser.add_argument("--group_size", default=128, type=int, help="group size")
    parser.add_argument("--bottle_ratio", default=0.5, type=float, help="bottle ratio")
    parser.add_argument("--feat_mult", default=2., type=float, help="feature multiplication")
    parser.add_argument("--act_layer", default="gelu", type=str, help="activation layer type")
    parser.add_argument("--attn_layer", default="se", type=str, help="attention layer type")
    parser.add_argument("--attn_kwargs", default=None, type=str, help="attention kwargs")

    return parser.parse_args()


def load_raw_data(args):
    id2image_name = dict()
    with open(os.path.join(args.data_dir, "item_info.jsonl"), "r", encoding="utf-8") as r:
        while True:
            line = r.readline()
            if not line:
                break
            d = json.loads(line.strip())
            id2image_name[d['item_id']] = f"{d['item_id']}.jpg" #d['item_image_name']
    logger.info(f"Finished load item info, size: {len(id2image_name)}")

    train_data = []
    if args.do_train:
        logger.info(f"Start loading train data")
        with open(os.path.join(args.data_dir, "item_train_pair.jsonl"), "r", encoding="utf-8") as r:
            i = 0
            while True:
                line = r.readline()
                if not line:
                    break
                d = json.loads(line.strip())
                item_label = int(d['item_label'])
                src_item_id = d['src_item_id']
                tgt_item_id = d['tgt_item_id']
                src_image_path = os.path.join(args.data_dir, "item_images_cropped", id2image_name[src_item_id])
                # src_img = Image.open(src_image_path)
                # src_img = src_img.convert("RGB")
                # src_input = transform_train_fn(src_img)
                tgt_image_path = os.path.join(args.data_dir, "item_images_cropped", id2image_name[tgt_item_id])
                # tgt_img = Image.open(tgt_image_path)
                # tgt_img = tgt_img.convert("RGB")
                # tgt_input = transform_train_fn(tgt_img)
                train_data.append((item_label, src_item_id, src_image_path, tgt_item_id, tgt_image_path))
                # train_data.append((item_label, src_item_id, src_input, tgt_item_id, tgt_input))
                # if i % N == 0:
                #     logger.info(f"Finished loading {i} train data")
                i += 1

    valid_data = []
    if args.do_eval:
        logger.info(f"Start loading eval data")
        with open(os.path.join(args.data_dir, "item_valid_pair.jsonl"), "r", encoding="utf-8") as r:
            i = 0
            while True:
                line = r.readline()
                if not line:
                    break
                d = json.loads(line.strip())
                item_label = int(d['item_label'])
                src_item_id = d['src_item_id']
                tgt_item_id = d['tgt_item_id']
                src_image_path = os.path.join(args.data_dir, "item_images", id2image_name[src_item_id])
                # src_img = Image.open(src_image_path)
                # src_img = src_img.convert("RGB")
                # src_input = transform_eval_fn(src_img)
                tgt_image_path = os.path.join(args.data_dir, "item_images", id2image_name[tgt_item_id])
                # tgt_img = Image.open(tgt_image_path)
                # tgt_img = tgt_img.convert("RGB")
                # tgt_input = transform_eval_fn(tgt_img)
                valid_data.append((item_label, src_item_id, src_image_path, tgt_item_id, tgt_image_path))
                # valid_data.append((item_label, src_item_id, src_input, tgt_item_id, tgt_input))
                # if i % N == 0:
                #     logger.info(f"Finished loading {i} eval data")
                i += 1

    test_data = []
    if args.do_pred:
        logger.info(f"Start loading test data")
        with open(os.path.join(args.data_dir, "item_test_pair.jsonl"), "r", encoding="utf-8") as r:
            i = 0
            while True:
                line = r.readline()
                if not line:
                    break
                d = json.loads(line.strip())
                item_label = 0
                src_item_id = d['src_item_id']
                tgt_item_id = d['tgt_item_id']
                src_image_path = os.path.join(args.data_dir, "item_images_cropped", id2image_name[src_item_id])
                # src_img = Image.open(src_image_path)
                # src_img = src_img.convert("RGB")
                # src_input = transform_eval_fn(src_img)
                tgt_image_path = os.path.join(args.data_dir, "item_images_cropped", id2image_name[tgt_item_id])
                # tgt_img = Image.open(tgt_image_path)
                # tgt_img = tgt_img.convert("RGB")
                # tgt_input = transform_eval_fn(tgt_img)
                test_data.append((item_label, src_item_id, src_image_path, tgt_item_id, tgt_image_path))
                # test_data.append((item_label, src_item_id, src_input, tgt_item_id, tgt_input))
                # if i % N == 0:
                #     logger.info(f"Finished loading {i} test data")
                i += 1

    del id2image_name

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
    # load model
    config = BertConfig.from_json_file(os.path.join(args.output_dir, args.config_file))
    config.loss_type = args.loss_type
    config.loss_margin = args.margin
    config.image_size = args.image_size
    image_encoder = timm.create_model(args.model_name, pretrained=True)
    if "nfnet" in args.model_name:
        # attn_kwargs = args.attn_kwargs if args.attn_kwargs is not None else dict(rd_ratio=0.5)
        # channels = tuple(int(c) for c in args.channels.split(","))
        # depths = tuple(int(c) for c in args.depths.split(","))
        # num_features = int(channels[-1] * args.feat_mult)
        # config = model_cfgs[args.model_name]
        # config = NfCfg(depths=depths, channels=channels, stem_type=args.stem_type, stem_chs=args.stem_chs,
        #                group_size=args.group_size, bottle_ratio=args.bottle_ratio, extra_conv=True,
        #                num_features=num_features, act_layer=args.act_layer, attn_layer=args.attn_layer,
        #                attn_kwargs=attn_kwargs)

        # model = NormFreeNet(config, num_classes=args.num_classes, in_chans=args.in_chans, global_pool=args.global_pool,
        #                     output_stride=args.stride, drop_rate=0., drop_path_rate=0)
        # if args.pretrained_model_path is not None:
        #     state_dict = torch.load(args.pretrained_model_path, map_location="cpu")
        #     model.load_state_dict(state_dict, strict=False)
        model = NFNetTwoTower(config, image_encoder)
    elif "vit" in args.model_name:
        # vit = ViT(config)
        # vit = Extractor(vit, device=device, layer_name="blocks", return_embeddings_only=True)
        # if args.pretrained_model_path is not None:
        #     vit.load_pretrained(args.pretrained_model_path)
        model = VitTwoTower(config, image_encoder)
    elif "resnet" in args.model_name:
        model = ResNetTwoTower(config, image_encoder)
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

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
        is_training = False if "vit" in args.model_name else True
        train_dataset = PairedImageDataset(train_data, input_size=args.image_size, is_training=is_training, hflip=args.hflip,
                                           color_jitter=args.color_jitter)
        train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                       collate_fn=collate_image)

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
        valid_dataset = PairedImageDataset(valid_data, input_size=args.image_size, is_training=False, hflip=args.hflip,
                                     color_jitter=args.color_jitter)
        valid_data_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                       collate_fn=collate_image)

    if device == "cuda":
        model.cuda()
        if args.do_train:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

    model_name = f"{args.model_name}-{args.data_version}"
    output_model_path = os.path.join(args.output_dir, model_name)
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
        logger.info("  Model name = %s", args.model_name)
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
                        src_inputs, tgt_inputs, labels = batch
                        output = model(src_inputs, tgt_inputs, labels)
                else:
                    src_inputs, tgt_inputs, labels = batch
                    output = model(src_inputs, tgt_inputs, labels)

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
                            src_inputs, tgt_inputs, labels = batch
                            output = model(src_inputs, tgt_inputs, labels)
                    else:
                        src_inputs, tgt_inputs, labels = batch
                        output = model(src_inputs, tgt_inputs, labels)

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
            output_model_file = os.path.join(output_model_path, f"image_finetune_epoch-{epoch}.bin")
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
                    src_inputs, tgt_inputs, labels = batch
                    output = model(src_inputs, tgt_inputs, labels)
            else:
                src_inputs, tgt_inputs, labels = batch
                output = model(src_inputs, tgt_inputs, labels)

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
        test_dataset = PairedImageDataset(test_data, input_size=args.image_size, is_training=False, hflip=args.hflip,
                                    color_jitter=args.color_jitter)
        test_data_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                      collate_fn=collate_image)

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
                        src_inputs, tgt_inputs, labels = batch
                        output = model(src_inputs, tgt_inputs, labels)
                else:
                    src_inputs, tgt_inputs, labels = batch
                    output = model(src_inputs, tgt_inputs, labels)

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
