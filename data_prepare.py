import sys
sys.path.insert(0, "/root/Code/yolov5")
sys.path.insert(0, "D:\\Code\\yolov5")
import os
import argparse
import json
import random
import jieba
import tensorpack.dataflow as td
import numpy as np
import torch
import timm
import traceback

from pathlib import Path
from PIL import Image
from timm.data.transforms_factory import create_transform
from timm.models.nfnet import model_cfgs

from src.models import NormFreeNet
from src.utils.logger import logger
# from torch_sparse import SparseTensor


RELATION_PADDING = "[PAD]"
RELATION_PADDING_ID = 0
# RELATION_INDUSTRY_NAME = "行业"
# RELATION_INDUSTRY_NAME_ID = 1
# RELATION_CATE_NAME = "品类"
# RELATION_CATE_NAME_ID = 2
# UNKNOW_ITEM = "/item/unknown"
# UNKNOW_VALUE = "/value/unknown"
# UNKNOW_ITEM_ID = 0
# UNKNOW_VALUE_ID = 1

CATE2YOLO_CLASS = {'手机保护套/壳': ['cell phone'],
                   '手机': ['cell phone'],
                   '空调': ['vase', 'cell phone', 'tv', 'microwave'],
                   '微波炉': ['micro wave'],
                   '电烤箱': ['microwave', 'oven'],
                   '剃须刀': ['parking meter', 'mouse', 'remote'],
                   '专业数码单反': ['cell phone', 'truck'],
                   '洗烘套装': ['cell phone', 'oven', 'traffic light'],
                   '养生壶/煎药壶': ['cup'],
                   '电热毯/电热垫/电热地毯': ['bed', 'suitcase', 'tie', 'handbag'],
                   '电热毯/水暖毯': ['bed'],
                   '智能音箱': ['cell phone', 'sports ball', 'laptop', 'vase', 'bottle'],
                   '茶吧机/泡茶机': ['refrigerator', 'oven'],
                   '洗地机': ['truck', 'toaster'],
                   '饮水机': ['refrigerator', 'cell phone', 'parking meter', 'laptop', 'microwave'],
                   '电磁炉/陶炉': ['cell phone', 'oven', 'toaster'],
                   '游戏电竞头戴耳机': ['scissors', 'cup', 'airplane', 'truck'],
                   '休闲裤': ['person'],
                   '毛衣': ['person'],
                   '帽子': ['person', 'kite'],
                   '马丁靴': ['person', 'handbag', 'suitcase'],
                   '金骏眉': ['bowl', 'dining table'],
                   '传统黄酒': ['bottle', 'vase'],
                   '驼奶及驼奶粉': ['book', 'cup', 'refrigerator', 'bottle'],
                   '果酒': ['bottle'],
                   '速溶咖啡': ['bottle', 'book'],
                   '牛仔裤': ['person', 'tie', 'handbag', 'suitcase'],
                   '休闲运动套装': ['person', 'bed'],
                   '中老年女装': ['person', 'vase'],
                   '半身裙': ['person', 'umbrella'],
                   '男士包袋': ['suitcase', 'handbag'],
                   '休闲皮鞋': ['handbag', 'person'],
                   '时尚套装': ['person'],
                   '蕾丝衫/雪纺衫': ['person', 'bed'],
                   '时尚休闲鞋': ['cake', 'person'],
                   '双肩背包': ['backpack', 'suitcase', 'handbag'],
                   '毛针织衫': ['person', 'tie'],
                   '围巾/丝巾/披肩': ['tie', 'person'],
                   '旗袍': ['person'],
                   '大码女装': ['person'],
                   '板鞋': ['handbag', 'suitcase', 'cake', 'surfboard', 'skateboard'],
                   '卫裤': ['person', 'handbag'],
                   '瑞士腕表': ['clock'],
                   '国产腕表': ['clock'],
                   '衬衫': ['person'],
                   '颈饰': ['person', 'cake', 'vase', 'sports ball', 'bottle', 'cup'],
                   '投资贵金属': ['clock', 'frisbee', 'teddy bear', 'horse', 'vase'],
                   '背心吊带': ['person', 'cake'],
                   '日韩腕表': ['clock'],
                   '钱包': ['suitcase', 'cell phone', 'handbag'],
                   '电动自行车': ['bicycle', 'motorcycle'],
                   '餐桌': ['dining table'],
                   '收纳箱': ['suitcase', 'vase', 'refrigerator', 'oven', 'surfboard', 'tv'],
                   '碗': ['bowl', 'cup'],
                   '炒锅': ['bowl'],
                   '鲜花速递(同城)': ['potted plant'],
                   '仓储货架': ['bench', 'bed'],
                   '垃圾桶': ['cup', 'toilet', 'refrigerator'],
                   '电脑椅': ['chair'],
                   '茶几': ['dining table'],
                   '化纤被': ['bed', 'person'],
                   '茶道/零配': ['vase', 'bottle', 'bowl', 'knife'],
                   '智能车机导航': ['cell phone', 'tv', 'car'],
                   '乳胶床垫': ['bed', 'suitcase', 'laptop'],
                   '普通坐便器': ['toilet', 'refrigerator'],
                   '狗狗': ['dog', 'teddy bear'],
                   '乳胶枕': ['keyboard',
                           'bed',
                           'remote',
                           'knife',
                           'surfboard',
                           'suitcase',
                           'cake'],
                   '弹簧床垫': ['bed', 'keyboard'],
                   '羽绒/羽毛被': ['bed'],
                   '桌布': ['dining table'],
                   '书桌': ['dining table'],
                   '椰棕床垫': ['bed', 'cake', 'keyboard'],
                   '电脑桌': ['dining table'],
                   '茶壶': ['vase', 'mouse'],
                   '投影机': ['toaster', 'microwave', 'car'],
                   '洗漱包': ['suitcase'],
                   '摩托车整车': ['truck', 'motorcycle'],
                   '护手霜': ['cup', 'book', 'bottle', 'frisbee', 'cell phonne'],
                   '贴片面膜': ['book', 'bottle'],
                   '隔离/妆前': ['bottle', 'toothbrush', 'refrigerator'],
                   '洗发水': ['bottle'],
                   '美甲工具': ['person', 'toothbrush', 'baseball bat'],
                   '润唇膏': ['cup', 'bottle'],
                   '男士面部乳霜': ['bottle', 'cell phone'],
                   '电动牙刷': ['toothbrush'],
                   '洗护套装': ['bottle', 'cup'],
                   '涂抹面膜': ['cup', 'book', 'bottle', 'vase'],
                   '化妆刷': ['knife',
                           'spoon',
                           'baseball bat',
                           'vase',
                           'toothbrush',
                           'scissors',
                           'book'],
                   '彩妆套装': ['suitcase'],
                   '身体乳/霜': ['bottle'],
                   '眼霜': ['cup', 'book', 'bottle', 'vase'],
                   '指甲彩妆': ['bottle', 'person'],
                   '私处保养': ['bottle', 'vase'],
                   '脱毛膏': ['bottle', 'book', 'cup'],
                   '男士护理套装': ['bottle', 'cell phone', 'microwave', 'refrigerator'],
                   '棉柔巾': ['book', 'remote'],
                   'KTV/卡拉OK音箱': ['tv'],
                   'DIY兼容机': ['microwave', 'traffic light'],
                   '自热火锅': ['bowl'],
                   '智能手环': ['cell phone'],
                   '智能手表': ['cell phone'],
                   '智能儿童手表': ['cell phone'],
                   '茶生壶/煎药壶': ['cup'],
                   '显示器': ['tv'],
                   '女士脱毛/剃毛器': ['cell phone', 'toothbrush', 'vase', 'tennis racket'],
                   '空气炸锅': ['oven', 'cell phone'],
                   '麦克风/话筒': ['toothbrush', 'parking meter'],
                   '空气净化器': ['refrigerator', 'cup'],
                   '净水器': ['bottle'],
                   '颈椎/腰椎按摩器': ['traffic light'],
                   '颈椎按摩器': ['scissors', 'mouse', 'traffic light', 'handbag'],
                   '键盘': ['keyboard'],
                   '加湿器': ['vase', 'refrigerator', 'cup', 'cell phone'],
                   '电子美容仪': ['vase', 'hair drier', 'scissors', 'toothbrush', 'cell phone'],
                   '电热水壶': ['cup', 'microwave', 'refrigerator'],
                   '电磁炉/掏炉': ['cell phone', 'toaster', 'oven'],
                   '电吹风': ['hair drier', 'motorcycle'],
                   '单反镜头': ['microwave', 'bottle', 'cell phone', 'book'],
                   '除螨仪': ['mouse', 'cell phone'],
                   '超声波迷你清洗机': ['cup'],
                   '笔记本电脑': ['laptop'],
                   '啤酒': ['bottle']}


def get_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", required=True, type=str, help="模型训练数据地址")
    parser.add_argument("--output_dir", required=True, type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--dtypes", default="train,valid,test", type=str, help="data types used")
    # PKGM pretraining
    parser.add_argument("--filter_method", default="freq", type=str, help="过滤relation的方法，2种取值：(1) freq: 基于最小频率，"
                                                                          "(2) topn：基于频率前n位")
    parser.add_argument("--min_freq", default=10, type=int, help="过滤relation的最小频率")
    parser.add_argument("--min_prop", default=0.5, type=float, help="过滤relation的最小比例（占该cate_name出现次数的比例）")
    parser.add_argument("--max_rank", default=20, type=int, help="过滤relation的最大排名")
    # text fintune
    parser.add_argument("--split_on_train", action="store_true", help="是否从训练数据拆分出train和valid")
    parser.add_argument("--prev_valid", default=None, type=str, help="之前的验证集，保证每次划分出的数据相同")
    parser.add_argument("--num_train_augment", default=0, type=int, help="需要增强的训练样本量")
    parser.add_argument("--num_neg", default=1, type=int, help="每条数据需要生成的负样本个数")
    parser.add_argument("--valid_proportion", default=0.25, type=float, help="从训练数据拆分时，valid的占比")
    parser.add_argument("--valid_pos_proportion", default=0.25, type=float, help="从训练数据拆分时，valid中正例的比例")
    # cv
    parser.add_argument("--only_image", action="store_true", help="if processing only image data")
    parser.add_argument("--with_image", action="store_true", help="whether to include pretrained image embedding")
    parser.add_argument("--cv_model_name", default="eca_nfnet_l1", type=str, help="cv pretrained model name")
    parser.add_argument("--finetuned", action="store_true", help="whether pretrained cv model is finetuned")
    parser.add_argument("--pretrained_model_path", default=None, type=str, help="model path for the finetuned cv model")
    parser.add_argument("--image_size", default=1000, type=int, help="raw image size, default: 1000*1000")
    parser.add_argument("--batch_size", default=256, type=int, help="cv model prediction batch size")
    parser.add_argument("--hflip", default=0.5, type=float, help="image transform: horizontal flip")
    parser.add_argument("--color_jitter", default=None, type=float, help="image transform: color jitter")
    parser.add_argument("--num_classes", default=2, type=int, help="number of label classes")
    parser.add_argument("--in_chans", default=3, type=int, help="number of in channels")
    parser.add_argument("--global_pool", default="avg", type=str, help="global pooling method")
    parser.add_argument("--stride", default=32, type=int, help="stride")
    parser.add_argument("--object_detection", action="store_true", help="whether to perform object detection on image")
    parser.add_argument("--code_path", default=None, type=str, help="yolov5 code path")
    parser.add_argument("--min_crop_ratio", default=0.1, type=float, help="minimum ratio of cropped area to original area")

    return parser.parse_args()


# def relation_filter(args):
#     # 统计relation数量
#     relation_count = dict()
#     ct = 0
#     for dtype in args.dtypes.split(","):
#         file_info = os.path.join(args.data_dir, f"item_{dtype}_info.jsonl")
#         with open(file_info, "r", encoding="utf-8") as r:
#             while True:
#                 line = r.readline()
#                 if not line:
#                     break
#                 d = json.loads(line.strip())
#                 cate_name = d['cate_name']
#
#                 # item pvs
#                 if 'item_pvs' in d:
#                     if cate_name not in relation_count:
#                         relation_count[cate_name] = dict()
#                     for item_pv in d['item_pvs'].replace("#", "").split(";"):
#                         try:
#                             relation_key, v = item_pv.split(":", maxsplit=1)
#                         except Exception:
#                             logger.warning(f"[Item Pv Split Error] {item_pv}")
#                             continue
#                         relation_key = relation_key.strip()
#                         if relation_key not in relation_count[cate_name]:
#                             relation_count[cate_name][relation_key] = 0
#                         relation_count[cate_name][relation_key] += 1
#                         ct += 1
#                 #         if ct > 1000:
#                 #             break
#                 # if ct > 1000:
#                 #     break
#     logger.info(f"[pkgm pretraining data] # cates: {len(relation_count)}")
#
#     # relation_count保存
#     # fo = open(os.path.join(args.output_dir, "relation_count.json"), "w", encoding="utf-8")
#     # json.dump(relation_count, fo, ensure_ascii=False)
#
#     # relation筛选
#     relation_include = set()
#     for cate, val_dict in relation_count.items():
#         # 方法1：根据最小出现次数筛选
#         if args.filter_method == "freq":
#             for relation, ct in val_dict.items():
#                 if ct >= args.min_freq:
#                     relation_include.add(relation)
#         # 方法2：根据出现次数top-n筛选
#         elif args.filter_method == "topn":
#             sorted_val_dict = {k: v for k, v in sorted(val_dict.items(), key=lambda item: item[1], reverse=True)}
#             for i, (cate, ct) in enumerate(sorted_val_dict.items()):
#                 if i >= args.max_rank:
#                     break
#                 relation_include.add(cate)
#
#     # relation_include.add(RELATION_CATE_NAME)
#     # relation_include.add(RELATION_INDUSTRY_NAME)
#     logger.info(f"[pkgm pretraining data] # relations included: {len(relation_include)}")
#
#     return relation_include, relation_count


def load_image_embedding(args):
    file_path = os.path.join(args.output_dir, "image_embedding.json")

    if os.path.isfile(file_path):
        img_emb_dict = json.load(open(file_path, "r", encoding="utf-8"))
    else:
        img_emb_dict = dict()
        # load pretrained cv model
        if args.finetuned:
            config = model_cfgs[args.cv_model_name]
            model = NormFreeNet(config, num_classes=args.num_classes, in_chans=args.in_chans, global_pool=args.global_pool,
                                output_stride=args.stride, drop_rate=0., drop_path_rate=0)
            state_dict = torch.load(args.pretrained_model_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            model = timm.create_model(args.cv_model_name, pretrained=True, num_classes=0)
        model.cuda()
        # image transform
        transform_fn = create_transform(input_size=args.image_size, is_training=False)
        # config = model.default_cfg
        # transform_fn = timm.data.transforms_factory.transforms_imagenet_eval(
        #     img_size=args.image_size,
        #     interpolation=config["interpolation"],
        #     mean=config["mean"],
        #     std=config["std"],
        #     crop_pct=config["crop_pct"],
        # )
        # read raw images
        img_emb_missing = np.zeros(model.num_features, dtype=np.float32).tolist()
        ct = 0
        ct_missing = 0
        ct_error = 0
        image_path = os.path.join(args.data_dir, f"item_images_cropped")
        file_info = os.path.join(args.data_dir, f"item_info.jsonl")
        ids = []
        batch = []
        with open(file_info, "r", encoding="utf-8") as r:
            while True:
                line = r.readline()
                if not line:
                    break
                d = json.loads(line.strip())
                item_id = d['item_id']
                img_name = f"{item_id}.jpg"
                # img_name = d.get('item_image_name', None)
                if img_name is None:
                    img_emb_dict[item_id] = img_emb_missing
                    logger.warning(f"[CV Error] {item_id}: image missing")
                    ct_missing += 1
                else:
                    try:
                        img = Image.open(os.path.join(image_path, img_name))
                        img = img.convert("RGB")
                        input_tensor = transform_fn(img).to("cuda")
                        batch.append(input_tensor)
                        ids.append(item_id)
                    except Exception as e:
                        img_emb_dict[item_id] = img_emb_missing
                        logger.warning(f"[CV Error] {item_id} {img_name}")
                        ct_error += 1
                    finally:
                        if len(batch) >= args.batch_size:
                            with torch.no_grad():
                                if args.finetuned:
                                    output = model.forward_features(torch.stack(batch))
                                    output = model.classifier.pool(output)
                                else:
                                    output = model(torch.stack(batch))
                            for iid, img_emb in zip(ids, output):
                                img_emb_dict[iid] = img_emb.cpu().detach().numpy().tolist()
                            batch = []
                            ids = []
                if ct % 10000 == 0:
                    logger.info(f"{ct} images processed")
                ct += 1
        if len(batch) > 0:
            with torch.no_grad():
                output = model(torch.stack(batch))
            for iid, img_emb in zip(ids, output):
                img_emb_dict[iid] = img_emb.cpu().detach().numpy().tolist()
            del batch
            del ids
        logger.info(f"Finished processing images! Total: {ct}, Missing: {ct_missing}, Error: {ct_error}")
        # save image embedding dict
        with open(file_path, "w", encoding="utf-8") as w:
            json.dump(img_emb_dict, w, ensure_ascii=False)

    logger.info(f"Finished loading image embedding dict! Length: {len(img_emb_dict)}")

    return img_emb_dict


def relation_filter(args):
    # 统计relation数量
    relation_count = dict()
    cate_count = dict()
    id_dict = dict()
    ct = 0
    file_info = os.path.join(args.data_dir, f"item_info.jsonl")
    with open(file_info, "r", encoding="utf-8") as r:
        while True:
            line = r.readline()
            if not line:
                break
            d = json.loads(line.strip())
            item_id = d['item_id']
            id_dict[item_id] = d
            cate_name = d['cate_name']
            if cate_name not in cate_count:
                cate_count[cate_name] = 0
            cate_count[cate_name] += 1
            # item pvs & sku pvs
            if cate_name not in relation_count:
                relation_count[cate_name] = dict()
            pvs = d.get('item_pvs', '').split("#;#") + d.get('sku_pvs', '').split("#;#")
            pvs_dict = dict()
            for pv in pvs:
                try:
                    relation_key, v = pv.split("#:#", maxsplit=1)
                except ValueError:
                    logger.warning(f"[Pv Split Error] {pv}")
                    continue
                relation_key = relation_key.strip()
                v = v.strip()
                if len(relation_key) <= 0 or len(v) <= 0:
                    continue
                if relation_key not in relation_count[cate_name]:
                    relation_count[cate_name][relation_key] = {"total": 0, "1-total": 0, "0-total": 0,
                                                               "1-same": 0, "0-diff": 0, "1-diff": 0,
                                                               "0-same": 0}
                if relation_key not in pvs_dict:
                    pvs_dict[relation_key] = set()
                    relation_count[cate_name][relation_key]['total'] += 1
                pvs_dict[relation_key].add(v)
                ct += 1
            d['pvs'] = pvs_dict
    logger.info(f"[pkgm pretraining data] # cates: {len(relation_count)}")

    # 统计重要relation: 从pair中提取公共的relation，选择label=1时取值相同而label=0时取值不同的relation为重要relation
    file_pairs = [
        os.path.join(args.data_dir, f"item_train_pair.jsonl"),
        # os.path.join(args.data_dir, f"item_valid_pair.jsonl")
    ]
    for file_pair in file_pairs:
        with open(file_pair, "r", encoding="utf-8") as r:
            while True:
                line = r.readline()
                if not line:
                    break
                d = json.loads(line.strip())
                src_item_id = d['src_item_id']
                tgt_item_id = d['tgt_item_id']
                item_label = d.get('item_label', "")
                d1 = id_dict[src_item_id]
                d2 = id_dict[tgt_item_id]
                if d1['cate_name'] != d2['cate_name']:
                    continue
                cate_name = d1['cate_name']
                pv1 = d1['pvs']
                pv2 = d1['pvs']
                rels1 = set(pv1.keys())
                rels2 = set(pv2.keys())
                rels = rels1.intersection(rels2)
                for rel in rels:
                    if item_label == "1":
                        if pv1[rel] == pv2[rel]:
                            relation_count[cate_name][rel]['1-same'] += 1
                        else:
                            relation_count[cate_name][rel]['1-diff'] += 1
                        relation_count[cate_name][rel]['1-total'] += 1
                    elif item_label == "0":
                        if pv1[rel] == pv2[rel]:
                            relation_count[cate_name][rel]['0-same'] += 1
                        else:
                            relation_count[cate_name][rel]['0-diff'] += 1
                        relation_count[cate_name][rel]['0-total'] += 1

    # relation筛选
    relation_include = set()
    for cate, val_dict in relation_count.items():
        # 方法1：根据最小出现次数筛选
        if args.filter_method == "freq":
            for relation, ct in val_dict.items():
                ct['0-diff-rate'] = 0.0
                ct['1-same-rate'] = 0.0
                if ct['total'] >= args.min_freq or ct['total'] >= cate_count[cate] * args.min_prop:
                    relation_include.add(relation)
                    ct['0-diff-rate'] = ct['0-diff'] / ct['0-total'] if ct['0-total'] > 0 else 0.0
                    ct['1-same-rate'] = ct['1-same'] / ct['1-total'] if ct['1-total'] > 0 else 0.0
        # 方法2：根据出现次数top-n筛选
        elif args.filter_method == "topn":
            sorted_val_dict = {k: v for k, v in sorted(val_dict.items(), key=lambda item: item[1], reverse=True)}
            for i, (cate, ct) in enumerate(sorted_val_dict.items()):
                if i >= args.max_rank:
                    break
                relation_include.add(cate)

    # relation_count保存
    # fo = open(os.path.join(args.output_dir, "relation_count.json"), "w", encoding="utf-8")
    # json.dump(relation_count, fo, ensure_ascii=False)

    # relation_include.add(RELATION_CATE_NAME)
    # relation_include.add(RELATION_INDUSTRY_NAME)
    logger.info(f"[pkgm pretraining data] # relations included: {len(relation_include)}")

    return id_dict, relation_include, relation_count


def post_processing(args):
    lef = {}
    rig = {}
    rellef = {}
    relrig = {}

    triple = open(os.path.join(args.output_dir, "train2id.txt"), "r")
    valid = open(os.path.join(args.output_dir, "valid2id.txt"), "r")
    test = open(os.path.join(args.output_dir, "test2id.txt"), "r")

    tot = (int)(triple.readline())
    for i in range(tot):
        content = triple.readline()
        h,t,r = content.strip().split()
        if not (h,r) in lef:
            lef[(h,r)] = []
        if not (r,t) in rig:
            rig[(r,t)] = []
        lef[(h,r)].append(t)
        rig[(r,t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

    tot = (int)(valid.readline())
    for i in range(tot):
        content = valid.readline()
        h,t,r = content.strip().split()
        if not (h,r) in lef:
            lef[(h,r)] = []
        if not (r,t) in rig:
            rig[(r,t)] = []
        lef[(h,r)].append(t)
        rig[(r,t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

    tot = (int)(test.readline())
    for i in range(tot):
        content = test.readline()
        h,t,r = content.strip().split()
        if not (h,r) in lef:
            lef[(h,r)] = []
        if not (r,t) in rig:
            rig[(r,t)] = []
        lef[(h,r)].append(t)
        rig[(r,t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

    test.close()
    valid.close()
    triple.close()

    f = open(os.path.join(args.output_dir, "type_constrain.txt"), "w")
    f.write("%d\n"%(len(rellef)))
    for i in rellef:
        f.write("%s\t%d"%(i,len(rellef[i])))
        for j in rellef[i]:
            f.write("\t%s"%(j))
        f.write("\n")
        f.write("%s\t%d"%(i,len(relrig[i])))
        for j in relrig[i]:
            f.write("\t%s"%(j))
        f.write("\n")
    f.close()

    rellef = {}
    totlef = {}
    relrig = {}
    totrig = {}
    # lef: (h, r)
    # rig: (r, t)
    for i in lef:
        if not i[1] in rellef:
            rellef[i[1]] = 0
            totlef[i[1]] = 0
        rellef[i[1]] += len(lef[i])
        totlef[i[1]] += 1.0

    for i in rig:
        if not i[0] in relrig:
            relrig[i[0]] = 0
            totrig[i[0]] = 0
        relrig[i[0]] += len(rig[i])
        totrig[i[0]] += 1.0

    s11=0
    s1n=0
    sn1=0
    snn=0
    f = open(os.path.join(args.output_dir, "test2id.txt"), "r")
    tot = (int)(f.readline())
    for i in range(tot):
        content = f.readline()
        h,t,r = content.strip().split()
        rign = rellef[r] / totlef[r]
        lefn = relrig[r] / totrig[r]
        if (rign < 1.5 and lefn < 1.5):
            s11+=1
        if (rign >= 1.5 and lefn < 1.5):
            s1n+=1
        if (rign < 1.5 and lefn >= 1.5):
            sn1+=1
        if (rign >= 1.5 and lefn >= 1.5):
            snn+=1
    f.close()

    f = open(os.path.join(args.output_dir, "test2id.txt"), "r")
    f11 = open(os.path.join(args.output_dir, "1-1.txt"), "w")
    f1n = open(os.path.join(args.output_dir, "1-n.txt"), "w")
    fn1 = open(os.path.join(args.output_dir, "n-1.txt"), "w")
    fnn = open(os.path.join(args.output_dir, "n-n.txt"), "w")
    fall = open(os.path.join(args.output_dir, "test2id_all.txt"), "w")
    tot = (int)(f.readline())
    fall.write("%d\n"%(tot))
    f11.write("%d\n"%(s11))
    f1n.write("%d\n"%(s1n))
    fn1.write("%d\n"%(sn1))
    fnn.write("%d\n"%(snn))
    for i in range(tot):
        content = f.readline()
        h,t,r = content.strip().split()
        rign = rellef[r] / totlef[r]
        lefn = relrig[r] / totrig[r]
        if (rign < 1.5 and lefn < 1.5):
            f11.write(content)
            fall.write("0"+"\t"+content)
        if (rign >= 1.5 and lefn < 1.5):
            f1n.write(content)
            fall.write("1"+"\t"+content)
        if (rign < 1.5 and lefn >= 1.5):
            fn1.write(content)
            fall.write("2"+"\t"+content)
        if (rign >= 1.5 and lefn >= 1.5):
            fnn.write(content)
            fall.write("3"+"\t"+content)
    fall.close()
    f.close()
    f11.close()
    f1n.close()
    fn1.close()
    fnn.close()


def pkgm_pretraining_data(args):
    # step 1: relation filter
    id_dict, relation_include, relation_count = relation_filter(args)

    # step 2: triplet, relation and entity id mapping
    # triplet
    triplets = set()
    # entity dict
    entity_dict = dict()
    # entity_dict[UNKNOW_ITEM] = UNKNOW_ITEM_ID
    # entity_dict[UNKNOW_VALUE] = UNKNOW_VALUE_ID
    # relation dict
    relation_dict = dict()
    relation_dict[RELATION_PADDING] = RELATION_PADDING_ID
    # relation_dict[RELATION_CATE_NAME] = RELATION_CATE_NAME_ID
    # relation_dict[RELATION_INDUSTRY_NAME] = RELATION_INDUSTRY_NAME_ID

    # entity_id = UNKNOW_VALUE_ID
    entity_id = -1
    relation_id = RELATION_PADDING_ID
    ct = 0
    # adjacency_matrix = np.zeros((230023, 230023), dtype=np.uint8)
    for item_id, d in id_dict.items():
        head_entity_key = f"/item/{item_id}"
        if head_entity_key not in entity_dict:
            entity_id += 1
            entity_dict[head_entity_key] = entity_id
        i = entity_dict[head_entity_key]

        # triplet - cate_name
        cate_name = d['cate_name']
        cate_id = d['cate_id']
        tail_entity_key = f"/value/{cate_name}-{cate_id}"
        if tail_entity_key not in entity_dict:
            entity_id += 1
            entity_dict[tail_entity_key] = entity_id
        # triplet = tuple((head_entity_key, RELATION_CATE_NAME, tail_entity_key))
        # triplets.add(triplet)
        # j = entity_dict[tail_entity_key]
        # adjacency_matrix[i, j] = 1
        # adjacency_matrix[j, i] = 1

        # triplet - industry_name
        industry_name = d['industry_name']
        tail_entity_key = f"/value/{industry_name}"
        if tail_entity_key not in entity_dict:
            entity_id += 1
            entity_dict[tail_entity_key] = entity_id
        # triplet = tuple((head_entity_key, RELATION_INDUSTRY_NAME, tail_entity_key))
        # triplets.add(triplet)
        # j = entity_dict[tail_entity_key]
        # adjacency_matrix[i, j] = 1
        # adjacency_matrix[j, i] = 1

        # item pvs & sku pvs
        pvs = d.get('pvs', dict())
        # pvs_ct = dict()
        for relation_key, vals in pvs.items():
            for v in vals:
                tail_entity_key = f"/value/{v}"
                if tail_entity_key not in entity_dict:
                    entity_id += 1
                    entity_dict[tail_entity_key] = entity_id
                # j = entity_dict[tail_entity_key]
                # adjacency_matrix[i, j] = 1
                # adjacency_matrix[j, i] = 1
            if relation_key not in relation_dict:
                relation_id += 1
                relation_dict[relation_key] = relation_id
            # triplet = tuple((entity_dict[head_entity_key], relation_dict[relation_key], entity_dict[tail_entity_key]))
            triplet = tuple((head_entity_key, relation_key, tail_entity_key))
            triplets.add(triplet)
            # pvs_ct[relation_key] = relation_count[cate_name][relation_key]
            ct += 1
        # sorted_pvs = [(k, pvs[k]) for k, _ in sorted(pvs_ct.items(), key=lambda item: (item[1]['0-diff-rate']+item[1]['1-same-rate'], item[1]['total'], item[0]), reverse=True)]
        # sorted_pvs = [(k, pvs[k]) for k, _ in sorted(pvs_ct.items(), key=lambda item: (item[1]['total'], item[0]), reverse=True)]
        # d['pvs'] = sorted_pvs
    logger.info(f"[pkgm pretraining data] # triplets: {len(triplets)}, # relations: {len(relation_dict)}, "
                f"# entities: {len(entity_dict)}")

    # saving
    file_entity2id = os.path.join(args.output_dir, "entity2id.txt")
    file_relation2id = os.path.join(args.output_dir, "relation2id.txt")
    # file_adj_t = os.path.join(args.output_dir, "adj_t.pt")
    with open(file_entity2id, "w", encoding="utf-8") as w:
        # w.write(str(len(entity_dict))+"\n")
        for entity_name, entity_id in entity_dict.items():
            w.write("\t".join((entity_name, str(entity_id)))+"\n")
    with open(file_relation2id, "w", encoding="utf-8") as w:
        # w.write(str(len(relation_dict))+"\n")
        for relation_name, relation_id in relation_dict.items():
            w.write("\t".join((relation_name, str(relation_id)))+"\n")
    # adj_t = SparseTensor.from_dense(torch.tensor(adjacency_matrix))
    # torch.save(adj_t, file_adj_t)

    # train, valid & test split
    triplets = list(triplets)
    random.shuffle(triplets)
    test_proportion = 0.0
    valid_proportion = 0.0
    test_split_index = int(len(triplets) * test_proportion)
    valid_split_index = test_split_index + int(len(triplets) * valid_proportion)
    triplets_test = triplets[:test_split_index]
    triplets_valid = triplets[test_split_index:valid_split_index]
    triplets_train = triplets[valid_split_index:]
    logger.info(f"[pkgm pretraining data] # train: {len(triplets_train)}, # valid: {len(triplets_valid)}, # test: {len(triplets_test)}")

    file_train2id = os.path.join(args.output_dir, "train2id.txt")
    with open(file_train2id, "w", encoding="utf-8") as w:
        # w.write(str(len(triplets_train))+"\n")
        for hid, rid, tid in triplets_train:
            w.write("\t".join((str(hid), str(rid), str(tid)))+"\n")
    file_valid2id = os.path.join(args.output_dir, "valid2id.txt")
    with open(file_valid2id, "w", encoding="utf-8") as w:
        # w.write(str(len(triplets_valid))+"\n")
        for hid, rid, tid in triplets_valid:
            w.write("\t".join((str(hid), str(rid), str(tid)))+"\n")
    file_test2id = os.path.join(args.output_dir, "test2id.txt")
    with open(file_test2id, "w", encoding="utf-8") as w:
        # w.write(str(len(triplets_test))+"\n")
        for hid, rid, tid in triplets_test:
            w.write("\t".join((str(hid), str(rid), str(tid)))+"\n")

    # step 3: post processing
    # post_processing(args)

    return id_dict, relation_count


def finetune_data(args, item_dict, relation_count, img_emb_dict):
    file_pair = os.path.join(args.data_dir, f"item_train_pair.jsonl")
    pairs_pos = []
    pairs_neg = []
    with open(file_pair, "r", encoding="utf-8") as r:
        while True:
            line = r.readline()
            if not line:
                break
            d = json.loads(line.strip())
            src_item_id = d['src_item_id']
            src_pvs = item_dict[src_item_id].get('pvs', dict())
            src_cate_name = item_dict[src_item_id]['cate_name']
            src_industry_name = item_dict[src_item_id]['industry_name']
            src_item_title = item_dict[src_item_id].get('title', "")
            src_item_title = " ".join(jieba.cut(src_item_title))
            tgt_item_id = d['tgt_item_id']
            tgt_pvs = item_dict[tgt_item_id].get('pvs', dict())
            tgt_cate_name = item_dict[tgt_item_id]['cate_name']
            tgt_industry_name = item_dict[tgt_item_id]['industry_name']
            tgt_item_title = item_dict[tgt_item_id].get('title', "")
            tgt_item_title = " ".join(jieba.cut(tgt_item_title))

            # src_pvs = ";".join([f"{k}:{','.join(vs)}" for k, vs in sorted(src_pvs.items(),
            #                                                              key=lambda item: (relation_count[src_cate_name][item[0]]['total'],
            #                                                                                relation_count[src_cate_name][item[0]]['0-diff-rate']+relation_count[src_cate_name][item[0]]['1-same-rate'],
            #                                                                                item[1]), reverse=True)])
            # tgt_pvs = ";".join([f"{k}:{','.join(vs)}" for k, vs in sorted(tgt_pvs.items(),
            #                                                              key=lambda item: (relation_count[tgt_cate_name][item[0]]['total'],
            #                                                                                relation_count[tgt_cate_name][item[0]]['0-diff-rate']+relation_count[tgt_cate_name][item[0]]['1-same-rate'],
            #                                                                                item[1]), reverse=True)])

            rels1 = set(src_pvs.keys())
            rels2 = set(tgt_pvs.keys())
            rels = rels1.intersection(rels2)

            # pv_pair = dict()
            # for rel in rels:
            #     is_same = "1" if pv1[rel] == pv2[rel] else "0"
            #     pv_pair[rel] = {"value": ":".join((rel, is_same)),
            #                     "ct": relation_count[src_cate_name][rel]['total'] + relation_count[tgt_cate_name][rel]['total']}
            # sorted_pv_pair = [v['value'] for k, v in sorted(pv_pair.items(), key=lambda item: item[1]['ct'], reverse=True)]
            # pv_pair = ";".join(sorted_pv_pair)

            pv1_union = {k: v for k, v in src_pvs.items() if k in rels}
            pv1_diff = {k: v for k, v in src_pvs.items() if k not in rels}
            # sorted_pv1_union = [f"{k}:{v}" for k, vs in sorted(pv1_union.items(),
            #     key=lambda item: (relation_count[src_cate_name][item[0]]['0-diff-rate']+relation_count[src_cate_name][item[0]]['1-same-rate']+relation_count[tgt_cate_name][item[0]]['0-diff-rate']+relation_count[tgt_cate_name][item[0]]['1-same-rate'],
            #                       relation_count[src_cate_name][item[0]]['total']+relation_count[tgt_cate_name][item[0]]['total'],
            #                       item[1]), reverse=True) for v in vs]
            # sorted_pv1_diff = [f"{k}:{v}" for k, vs in sorted(pv1_diff.items(),
            #     key=lambda item: (relation_count[src_cate_name][item[0]]['0-diff-rate']+relation_count[src_cate_name][item[0]]['1-same-rate'],
            #                       relation_count[src_cate_name][item[0]]['total'],
            #                       item[1]), reverse=True) for v in vs]
            sorted_pv1_union = [f"{k}:{','.join(vs)}" for k, vs in sorted(pv1_union.items(),
                                                               key=lambda item: (relation_count[src_cate_name][item[0]]['total']+relation_count[tgt_cate_name][item[0]]['total'],
                                                                                 relation_count[src_cate_name][item[0]]['0-diff-rate']+relation_count[src_cate_name][item[0]]['1-same-rate']+relation_count[tgt_cate_name][item[0]]['0-diff-rate']+relation_count[tgt_cate_name][item[0]]['1-same-rate'],
                                                                                 item[1]), reverse=True)]
            sorted_pv1_diff = [f"{k}:{','.join(vs)}" for k, vs in sorted(pv1_diff.items(),
                                                              key=lambda item: (relation_count[src_cate_name][item[0]]['total'],
                                                                                relation_count[src_cate_name][item[0]]['0-diff-rate']+relation_count[src_cate_name][item[0]]['1-same-rate'],
                                                                                item[1]), reverse=True)]
            # src_pvs = ";".join([f"{RELATION_INDUSTRY_NAME}:{src_industry_name}", f"{RELATION_CATE_NAME}:{src_cate_name}"]+sorted_pv1_union+sorted_pv1_diff)
            src_pvs = ";".join(sorted_pv1_union+sorted_pv1_diff)

            pv2_union = {k: v for k, v in tgt_pvs.items() if k in rels}
            pv2_diff = {k: v for k, v in tgt_pvs.items() if k not in rels}
            # sorted_pv2_union = [f"{k}:{v}" for k, vs in sorted(pv2_union.items(),
            #     key=lambda item: (relation_count[src_cate_name][item[0]]['0-diff-rate']+relation_count[src_cate_name][item[0]]['1-same-rate']+relation_count[tgt_cate_name][item[0]]['0-diff-rate']+relation_count[tgt_cate_name][item[0]]['1-same-rate'],
            #                       relation_count[src_cate_name][item[0]]['total']+relation_count[tgt_cate_name][item[0]]['total'],
            #                       item[1]), reverse=True) for v in vs]
            # sorted_pv2_diff = [f"{k}:{v}" for k, vs in sorted(pv2_diff.items(),
            #     key=lambda item: (relation_count[tgt_cate_name][item[0]]['0-diff-rate']+relation_count[tgt_cate_name][item[0]]['1-same-rate'],
            #                       relation_count[tgt_cate_name][item[0]]['total'],
            #                       item[1]), reverse=True) for v in vs]
            sorted_pv2_union = [f"{k}:{','.join(vs)}" for k, vs in sorted(pv2_union.items(),
                                                               key=lambda item: (relation_count[src_cate_name][item[0]]['total']+relation_count[tgt_cate_name][item[0]]['total'],
                                                                                 relation_count[src_cate_name][item[0]]['0-diff-rate']+relation_count[src_cate_name][item[0]]['1-same-rate']+relation_count[tgt_cate_name][item[0]]['0-diff-rate']+relation_count[tgt_cate_name][item[0]]['1-same-rate'],
                                                                                 item[1]), reverse=True)]
            sorted_pv2_diff = [f"{k}:{','.join(vs)}" for k, vs in sorted(pv2_diff.items(),
                                                              key=lambda item: (relation_count[tgt_cate_name][item[0]]['total'],
                                                                                relation_count[tgt_cate_name][item[0]]['0-diff-rate']+relation_count[tgt_cate_name][item[0]]['1-same-rate'],
                                                                                item[1]), reverse=True)]
            # tgt_pvs = ";".join([f"{RELATION_INDUSTRY_NAME}:{tgt_industry_name}", f"{RELATION_CATE_NAME}:{tgt_cate_name}"]+sorted_pv2_union+sorted_pv2_diff)
            tgt_pvs = ";".join(sorted_pv2_union+sorted_pv2_diff)

            if img_emb_dict is not None:
                src_img_emb = ",".join([str(emb) for emb in img_emb_dict[src_item_id]])
                tgt_img_emb = ",".join([str(emb) for emb in img_emb_dict[tgt_item_id]])
            else:
                src_img_emb = None
                tgt_img_emb = None

            item_label = d['item_label']
            if item_label == "1":
                if src_img_emb is None or tgt_img_emb is None:
                    pairs_pos.append((item_label, src_item_id, src_item_title, src_pvs,
                                      tgt_item_id, tgt_item_title, tgt_pvs))
                    # pairs_pos.append((item_label, src_item_id, src_item_title, tgt_item_id,
                #                   tgt_item_title, pv_pair))
                else:
                    pairs_pos.append((item_label, src_item_id, src_item_title, src_pvs, src_img_emb,
                          tgt_item_id, tgt_item_title, tgt_pvs, tgt_img_emb))

            elif item_label == "0":
                if src_img_emb is None or tgt_img_emb is None:
                    pairs_neg.append((item_label, src_item_id, src_item_title, src_pvs,
                          tgt_item_id, tgt_item_title, tgt_pvs))
                else:
                    pairs_neg.append((item_label, src_item_id, src_item_title, src_pvs, src_img_emb,
                          tgt_item_id, tgt_item_title, tgt_pvs, tgt_img_emb))
                # pairs_neg.append((item_label, src_item_id, src_item_title, tgt_item_id,
                #                   tgt_item_title, pv_pair))

    if args.split_on_train:
        if args.prev_valid is None:
            length = len(pairs_pos) + len(pairs_neg)
            # random.shuffle(pairs_pos)
            # random.shuffle(pairs_neg)
            # idx_pos = int(length * args.valid_proportion * args.valid_pos_proportion)
            # idx_neg = int(length * args.valid_proportion * (1-args.valid_pos_proportion))
            # pairs_valid_pos = pairs_pos[:idx_pos]
            # pairs_train_pos = pairs_pos[idx_pos:]
            # pairs_valid_neg = pairs_neg[:idx_neg]
            # pairs_train_neg = pairs_neg[idx_neg:]
            # pairs_train = pairs_train_pos + pairs_train_neg
            # pairs_valid = pairs_valid_pos + pairs_valid_neg
            pairs = pairs_pos+pairs_neg
            random.shuffle(pairs)
            idx = int(length*args.valid_proportion)
            pairs_valid = pairs[:idx]
            pairs_train = pairs[idx:]
        else:
            if img_emb_dict is None:
                pairs_dict = {f"{p[1]}-{p[4]}": p for p in pairs_pos+pairs_neg}
            else:
                pairs_dict = {f"{p[1]}-{p[5]}": p for p in pairs_pos+pairs_neg}
            pairs_valid = []
            with open(args.prev_valid, "r", encoding="utf-8") as r:
                while True:
                    line = r.readline()
                    if not line:
                        break
                    items = line.strip("\n").split("\t")
                    src_item_id = items[1]
                    tgt_item_id = items[4]
                    key = f"{src_item_id}-{tgt_item_id}"
                    pairs_valid.append(pairs_dict.pop(key))
            pairs_train = [p for _, p in pairs_dict.items()]
        train_data_file = "finetune_train_train.tsv"
        valid_data_file = "finetune_train_valid.tsv"
        easy_train_data_file = "finetune_train_train_easy.tsv"
        pairs_train_pos = [pair for pair in pairs_train if pair[0] == "1"]
    else:
        pairs_train = pairs_pos+pairs_neg
        pairs_train_pos = pairs_pos
        file_pair = os.path.join(args.data_dir, f"item_test_pair.jsonl")
        pairs_valid = []
        train_data_file = "finetune_train.tsv"
        valid_data_file = "finetune_test.tsv"
        # easy_train_data_file = "finetune_train_easy.tsv"
        with open(file_pair, "r", encoding="utf-8") as r:
            while True:
                line = r.readline()
                if not line:
                    break
                d = json.loads(line.strip())
                # item 1
                src_item_id = d['src_item_id']
                src_pvs = item_dict[src_item_id].get('pvs', dict())
                src_cate_name = item_dict[src_item_id]['cate_name']
                src_industry_name = item_dict[src_item_id]['industry_name']
                src_item_title = item_dict[src_item_id].get('title', "")
                src_item_title = " ".join(jieba.cut(src_item_title))
                # item 2
                tgt_item_id = d['tgt_item_id']
                tgt_pvs = item_dict[tgt_item_id].get('pvs', dict())
                tgt_cate_name = item_dict[tgt_item_id]['cate_name']
                tgt_industry_name = item_dict[tgt_item_id]['industry_name']
                tgt_item_title = item_dict[tgt_item_id].get('title', "")
                tgt_item_title = " ".join(jieba.cut(tgt_item_title))

                # src_pvs = ";".join([f"{k}:{','.join(vs)}" for k, vs in sorted(src_pvs.items(),
                #                                                               key=lambda item: (relation_count[src_cate_name][item[0]]['total'],
                #                                                                                 relation_count[src_cate_name][item[0]]['0-diff-rate']+relation_count[src_cate_name][item[0]]['1-same-rate'],
                #                                                                                 item[1]), reverse=True)])
                # tgt_pvs = ";".join([f"{k}:{','.join(vs)}" for k, vs in sorted(tgt_pvs.items(),
                #                                                               key=lambda item: (relation_count[tgt_cate_name][item[0]]['total'],
                #                                                                                 relation_count[tgt_cate_name][item[0]]['0-diff-rate']+relation_count[tgt_cate_name][item[0]]['1-same-rate'],
                #                                                                                 item[1]), reverse=True)])

                rels1 = set(src_pvs.keys())
                rels2 = set(tgt_pvs.keys())
                rels = rels1.intersection(rels2)

                # pv_pair = dict()
                # for rel in rels:
                #     is_same = "1" if pv1[rel] == pv2[rel] else "0"
                #     pv_pair[rel] = {"value": ":".join((rel, is_same)),
                #                     "ct": relation_count[src_cate_name][rel]['total'] + relation_count[tgt_cate_name][rel]['total']}
                # sorted_pv_pair = [v['value'] for k, v in sorted(pv_pair.items(), key=lambda item: item[1]['ct'], reverse=True)]
                # pv_pair = ";".join(sorted_pv_pair)

                pv1_union = {k: v for k, v in src_pvs.items() if k in rels}
                pv1_diff = {k: v for k, v in src_pvs.items() if k not in rels}
                # sorted_pv1_union = [f"{k}:{v}" for k, vs in sorted(pv1_union.items(),
                #     key=lambda item: (relation_count[src_cate_name][item[0]]['0-diff-rate']+relation_count[src_cate_name][item[0]]['1-same-rate']+relation_count[tgt_cate_name][item[0]]['0-diff-rate']+relation_count[tgt_cate_name][item[0]]['1-same-rate'],
                #                       relation_count[src_cate_name][item[0]]['total']+relation_count[tgt_cate_name][item[0]]['total'],
                #                       item[1]), reverse=True) for v in vs]
                # sorted_pv1_diff = [f"{k}:{v}" for k, vs in sorted(pv1_diff.items(),
                #     key=lambda item: (relation_count[src_cate_name][item[0]]['0-diff-rate']+relation_count[src_cate_name][item[0]]['1-same-rate'],
                #                       relation_count[src_cate_name][item[0]]['total'],
                #                       item[1]), reverse=True) for v in vs]
                sorted_pv1_union = [f"{k}:{','.join(vs)}" for k, vs in sorted(pv1_union.items(),
                                                                   key=lambda item: (relation_count[src_cate_name][item[0]]['total']+relation_count[tgt_cate_name][item[0]]['total'],
                                                                                     relation_count[src_cate_name][item[0]]['0-diff-rate']+relation_count[src_cate_name][item[0]]['1-same-rate']+relation_count[tgt_cate_name][item[0]]['0-diff-rate']+relation_count[tgt_cate_name][item[0]]['1-same-rate'],
                                                                                     item[1]), reverse=True)]
                sorted_pv1_diff = [f"{k}:{','.join(vs)}" for k, vs in sorted(pv1_diff.items(),
                                                                  key=lambda item: (relation_count[src_cate_name][item[0]]['total'],
                                                                                    relation_count[src_cate_name][item[0]]['0-diff-rate']+relation_count[src_cate_name][item[0]]['1-same-rate'],
                                                                                    item[1]), reverse=True)]
                # src_pvs = ";".join([f"{RELATION_INDUSTRY_NAME}:{src_industry_name}", f"{RELATION_CATE_NAME}:{src_cate_name}"]+sorted_pv1_union+sorted_pv1_diff)
                src_pvs = ";".join(sorted_pv1_union+sorted_pv1_diff)

                pv2_union = {k: v for k, v in tgt_pvs.items() if k in rels}
                pv2_diff = {k: v for k, v in tgt_pvs.items() if k not in rels}
                # sorted_pv2_union = [f"{k}:{v}" for k, vs in sorted(pv2_union.items(),
                #     key=lambda item: (relation_count[src_cate_name][item[0]]['0-diff-rate']+relation_count[src_cate_name][item[0]]['1-same-rate']+relation_count[tgt_cate_name][item[0]]['0-diff-rate']+relation_count[tgt_cate_name][item[0]]['1-same-rate'],
                #                       relation_count[src_cate_name][item[0]]['total']+relation_count[tgt_cate_name][item[0]]['total'],
                #                       item[1]), reverse=True) for v in vs]
                # sorted_pv2_diff = [f"{k}:{v}" for k, vs in sorted(pv2_diff.items(),
                #     key=lambda item: (relation_count[tgt_cate_name][item[0]]['0-diff-rate']+relation_count[tgt_cate_name][item[0]]['1-same-rate'],
                #                       relation_count[tgt_cate_name][item[0]]['total'],
                #                       item[1]), reverse=True) for v in vs]
                sorted_pv2_union = [f"{k}:{','.join(vs)}" for k, vs in sorted(pv2_union.items(),
                                                                   key=lambda item: (relation_count[src_cate_name][item[0]]['total']+relation_count[tgt_cate_name][item[0]]['total'],
                                                                                     relation_count[src_cate_name][item[0]]['0-diff-rate']+relation_count[src_cate_name][item[0]]['1-same-rate']+relation_count[tgt_cate_name][item[0]]['0-diff-rate']+relation_count[tgt_cate_name][item[0]]['1-same-rate'],
                                                                                     item[1]), reverse=True)]
                sorted_pv2_diff = [f"{k}:{','.join(vs)}" for k, vs in sorted(pv2_diff.items(),
                                                                  key=lambda item: (relation_count[tgt_cate_name][item[0]]['total'],
                                                                                    relation_count[tgt_cate_name][item[0]]['0-diff-rate']+relation_count[tgt_cate_name][item[0]]['1-same-rate'],
                                                                                    item[1]), reverse=True)]
                # tgt_pvs = ";".join([f"{RELATION_INDUSTRY_NAME}:{tgt_industry_name}", f"{RELATION_CATE_NAME}:{tgt_cate_name}"]+sorted_pv2_union+sorted_pv2_diff)
                tgt_pvs = ";".join(sorted_pv2_union+sorted_pv2_diff)

                if img_emb_dict is None:
                    src_img_emb = None
                    tgt_img_emb = None
                else:
                    src_img_emb = ",".join([str(emb) for emb in img_emb_dict[src_item_id]])
                    tgt_img_emb = ",".join([str(emb) for emb in img_emb_dict[tgt_item_id]])
                item_label = d.get('item_label', "0")

                if src_img_emb is None or tgt_img_emb is None:
                    pairs_valid.append((item_label, src_item_id, src_item_title, src_pvs,
                                        tgt_item_id, tgt_item_title, tgt_pvs))
                else:
                    pairs_valid.append((item_label, src_item_id, src_item_title, src_pvs, src_img_emb,
                                        tgt_item_id, tgt_item_title, tgt_pvs, tgt_img_emb))
                # pairs_valid.append((item_label, src_item_id, src_item_title, tgt_item_id,
                #                     tgt_item_title, pv_pair))

    # augment training data with pairs from different cate_name
    keys = list(item_dict.keys())
    keys_selected = random.sample(keys, args.num_train_augment)
    pairs_easy_neg = []
    for id1 in keys_selected:
        d1 = item_dict[id1]
        cate_name1 = d1['cate_name']
        industry_name1 = d1['industry_name']
        title1 = d1.get('title', '')
        pv1 = d1.get('pvs', dict())
        ids = set()
        while len(ids) < args.num_neg:
            id2 = random.choice(keys)
            d2 = item_dict[id2]
            cate_name2 = d2['cate_name']
            industry_name2 = d2['industry_name']
            if cate_name1 == cate_name2 or id2 == id1:
                continue
            title2 = d2.get('title', '')
            pv2 = d2.get('pvs', dict())

            pvs1 = ";".join([f"{k}:{','.join(vs)}" for k, vs in sorted(pv1.items(),
                                                                          key=lambda item: (relation_count[cate_name1][item[0]]['total'],
                                                                                            relation_count[cate_name1][item[0]]['0-diff-rate']+relation_count[cate_name1][item[0]]['1-same-rate'],
                                                                                            item[1]), reverse=True)])
            pvs2 = ";".join([f"{k}:{','.join(vs)}" for k, vs in sorted(pv2.items(),
                                                                          key=lambda item: (relation_count[cate_name2][item[0]]['total'],
                                                                                            relation_count[cate_name2][item[0]]['0-diff-rate']+relation_count[cate_name2][item[0]]['1-same-rate'],
                                                                                            item[1]), reverse=True)])
            # rels1 = set(pv1.keys())
            # rels2 = set(pv2.keys())
            # rels = rels1.intersection(rels2)
            # # pv_pair = dict()
            # # for rel in rels:
            # #     is_same = "1" if pv1[rel] == pv2[rel] else "0"
            # #     pv_pair[rel] = {"value": ":".join((rel, is_same)),
            # #                     "ct": relation_count[cate_name1][rel]['total'] + relation_count[cate_name2][rel]['total']}
            # # sorted_pv_pair = [v['value'] for k, v in sorted(pv_pair.items(), key=lambda item: item[1]['ct'], reverse=True)]
            # # pv_pair = ";".join(sorted_pv_pair)
            # pv1_union = {k: v for k, v in pv1.items() if k in rels}
            # pv1_diff = {k: v for k, v in pv1.items() if k not in rels}
            # pv2_union = {k: v for k, v in pv2.items() if k in rels}
            # pv2_diff = {k: v for k, v in pv2.items() if k not in rels}
            # # sorted_pv1_union = [f"{k}:{v}" for k, vs in sorted(pv1_union.items(),
            # #     key=lambda item: (relation_count[cate_name1][item[0]]['0-diff-rate']+relation_count[cate_name1][item[0]]['1-same-rate']+relation_count[cate_name2][item[0]]['0-diff-rate']+relation_count[cate_name2][item[0]]['1-same-rate'],
            # #                       relation_count[cate_name1][item[0]]['total']+relation_count[cate_name2][item[0]]['total'],
            # #                       item[1]), reverse=True) for v in vs]
            # # sorted_pv1_diff = [f"{k}:{v}" for k, vs in sorted(pv1_diff.items(),
            # #     key=lambda item: (relation_count[cate_name1][item[0]]['0-diff-rate']+relation_count[cate_name1][item[0]]['1-same-rate'],
            # #                       relation_count[cate_name1][item[0]]['total'],
            # #                       item[1]), reverse=True) for v in vs]
            # # sorted_pv2_union = [f"{k}:{v}" for k, vs in sorted(pv2_union.items(),
            # #     key=lambda item: (relation_count[cate_name1][item[0]]['0-diff-rate']+relation_count[cate_name1][item[0]]['1-same-rate']+relation_count[cate_name2][item[0]]['0-diff-rate']+relation_count[cate_name2][item[0]]['1-same-rate'],
            # #                       relation_count[cate_name1][item[0]]['total']+relation_count[cate_name2][item[0]]['total'],
            # #                       item[1]), reverse=True) for v in vs]
            # # sorted_pv2_diff = [f"{k}:{v}" for k, vs in sorted(pv2_diff.items(),
            # #     key=lambda item: (relation_count[cate_name2][item[0]]['0-diff-rate']+relation_count[cate_name2][item[0]]['1-same-rate'],
            # #                       relation_count[cate_name2][item[0]]['total'],
            # #                       item[1]), reverse=True) for v in vs]
            # sorted_pv1_union = [f"{k}:{v}" for k, vs in sorted(pv1_union.items(),
            #                                                    key=lambda item: (relation_count[cate_name1][item[0]]['total']+relation_count[cate_name2][item[0]]['total'],
            #                                                                      relation_count[cate_name1][item[0]]['0-diff-rate']+relation_count[cate_name1][item[0]]['1-same-rate']+relation_count[cate_name2][item[0]]['0-diff-rate']+relation_count[cate_name2][item[0]]['1-same-rate'],
            #                                                                      item[1]), reverse=True) for v in vs]
            # sorted_pv1_diff = [f"{k}:{v}" for k, vs in sorted(pv1_diff.items(),
            #                                                   key=lambda item: (relation_count[cate_name1][item[0]]['total'],
            #                                                                     relation_count[cate_name1][item[0]]['0-diff-rate']+relation_count[cate_name1][item[0]]['1-same-rate'],
            #                                                                     item[1]), reverse=True) for v in vs]
            # sorted_pv2_union = [f"{k}:{v}" for k, vs in sorted(pv2_union.items(),
            #                                                    key=lambda item: (relation_count[cate_name1][item[0]]['total']+relation_count[cate_name2][item[0]]['total'],
            #                                                                      relation_count[cate_name1][item[0]]['0-diff-rate']+relation_count[cate_name1][item[0]]['1-same-rate']+relation_count[cate_name2][item[0]]['0-diff-rate']+relation_count[cate_name2][item[0]]['1-same-rate'],
            #                                                                      item[1]), reverse=True) for v in vs]
            # sorted_pv2_diff = [f"{k}:{v}" for k, vs in sorted(pv2_diff.items(),
            #                                                   key=lambda item: (relation_count[cate_name2][item[0]]['total'],
            #                                                                     relation_count[cate_name2][item[0]]['0-diff-rate']+relation_count[cate_name2][item[0]]['1-same-rate'],
            #                                                                     item[1]), reverse=True) for v in vs]
            # # pvs1 = ";".join([f"{RELATION_INDUSTRY_NAME}:{industry_name1}", f"{RELATION_CATE_NAME}:{cate_name1}"]+sorted_pv1_union+sorted_pv1_diff)
            # # pvs2 = ";".join([f"{RELATION_INDUSTRY_NAME}:{industry_name2}", f"{RELATION_CATE_NAME}:{cate_name2}"]+sorted_pv2_union+sorted_pv2_diff)
            # pvs1 = ";".join(sorted_pv1_union+sorted_pv1_diff)
            # pvs2 = ";".join(sorted_pv2_union+sorted_pv2_diff)

            if img_emb_dict is None:
                img_emb1 = None
                img_emb2 = None
            else:
                img_emb1 = ",".join([str(emb) for emb in img_emb_dict[id1]])
                img_emb2 = ",".join([str(emb) for emb in img_emb_dict[id2]])
            item_label = "0"

            if img_emb1 is None or img_emb2 is None:
                # pairs_easy_neg.append((item_label, id1, title1, pvs1,
                #                     id2, title2, pvs2))
                pairs_train.append((item_label, id1, title1, pvs1,
                                       id2, title2, pvs2))
            else:
                # pairs_easy_neg.append((item_label, id1, title1, pvs1, img_emb1,
                #                     id2, title2, pvs2, img_emb2))
                pairs_train.append((item_label, id1, title1, pvs1, img_emb1,
                                       id2, title2, pvs2, img_emb2))
            ids.add(id2)

    logger.info(f"[finetune data] # train: {len(pairs_train)}, # valid: {len(pairs_valid)}")

    with open(os.path.join(args.output_dir, train_data_file), "w", encoding="utf-8") as w:
        random.shuffle(pairs_train)
        for pair in pairs_train:
            w.write("\t".join(pair)+'\n')

    with open(os.path.join(args.output_dir, valid_data_file), "w", encoding="utf-8") as w:
        for pair in pairs_valid:
            w.write("\t".join(pair)+'\n')

    # if len(pairs_easy_neg) > 0:
    #     with open(os.path.join(args.output_dir, easy_train_data_file), "w", encoding="utf-8") as w:
    #         pairs_train_easy = pairs_train_pos + pairs_easy_neg
    #         random.shuffle(pairs_train_easy)
    #         for pair in pairs_train_easy:
    #             w.write("\t".join(pair)+'\n')


def load_raw_data(args):
    id2image_name = dict()
    with open(os.path.join(args.data_dir, "item_info.jsonl"), "r", encoding="utf-8") as r:
        while True:
            line = r.readline()
            if not line:
                break
            d = json.loads(line.strip())
            id2image_name[d['item_id']] = d['item_image_name']
    logger.info(f"Finished load item info, size: {len(id2image_name)}")

    # transform_train_fn = create_transform(input_size=args.image_size, is_training=True,
    #                                       hflip=args.hflip, color_jitter=args.color_jitter)
    # transform_eval_fn = create_transform(input_size=args.image_size, is_training=False,
    #                                      hflip=args.hflip, color_jitter=args.color_jitter)
    # N = 100
    train_data = []
    if 'train' in args.dtypes:
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
                src_image_path = os.path.join(args.data_dir, "item_images", id2image_name[src_item_id])
                # src_img = Image.open(src_image_path)
                # src_img = src_img.convert("RGB")
                # src_input = transform_train_fn(src_img)
                tgt_image_path = os.path.join(args.data_dir, "item_images", id2image_name[tgt_item_id])
                # tgt_img = Image.open(tgt_image_path)
                # tgt_img = tgt_img.convert("RGB")
                # tgt_input = transform_train_fn(tgt_img)
                train_data.append((item_label, src_item_id, src_image_path, tgt_item_id, tgt_image_path))
                # train_data.append((item_label, src_item_id, src_input, tgt_item_id, tgt_input))
                # if i % N == 0:
                #     logger.info(f"Finished loading {i} train data")
                #     # break
                i += 1

    valid_data = []
    if 'valid' in args.dtypes:
        logger.info(f"Start loading eval data")
        with open(os.path.join(args.data_dir, "item_valid_pair.jsonl"), "r", encoding="utf-8") as r:
            i = 0
            while True:
                line = r.readline()
                if not line:
                    break
                d = json.loads(line.strip())
                item_label = 0
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
                # valid_data.append((item_label, src_item_id, src_input, tgt_item_id, tgt_input))
                valid_data.append((item_label, src_item_id, src_image_path, tgt_item_id, tgt_image_path))
                # if i % N == 0:
                #     logger.info(f"Finished loading {i} eval data")
                #     # break
                i += 1

    test_data = []
    if 'test' in args.dtypes:
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
                src_image_path = os.path.join(args.data_dir, "item_images", id2image_name[src_item_id])
                # src_img = Image.open(src_image_path)
                # src_img = src_img.convert("RGB")
                # src_input = transform_eval_fn(src_img)
                tgt_image_path = os.path.join(args.data_dir, "item_images", id2image_name[tgt_item_id])
                # tgt_img = Image.open(tgt_image_path)
                # tgt_img = tgt_img.convert("RGB")
                # tgt_input = transform_eval_fn(tgt_img)
                # test_data.append((item_label, src_item_id, src_input, tgt_item_id, tgt_input))
                test_data.append((item_label, src_item_id, src_image_path, tgt_item_id, tgt_image_path))
                # if i % N == 0:
                #     logger.info(f"Finished loading {i} test data")
                i += 1

    del id2image_name

    return train_data, valid_data, test_data


class ImageData(td.RNGDataFlow):
    """
    """
    def __init__(self, args, datas, file_type, transform_fn, id2image_name, shuffle=True):
        """
        Same as in :class:`ILSVRC12`.
        """
        self.lines = []
        i = 0
        logger.info(f"Start loading {file_type} data")

        # with open(os.path.join(args.data_dir, file_pair), "r", encoding="utf-8") as r:
        #     while True:
        #         line = r.readline()
        #         if not line:
        #             break
        #         d = json.loads(line.strip())
        #         item_label = d.get('item_label', None)
        #         if item_label is None:
        #             item_label = 0
        #         else:
        #             item_label = int(item_label)
        #         src_item_id = d['src_item_id']
        #         tgt_item_id = d['tgt_item_id']
        #         try:
        #             src_image_path = os.path.join(image_dir, id2image_name[src_item_id])
        #             src_img = Image.open(src_image_path)
        #             src_img = src_img.convert("RGB")
        #             src_input = transform_fn(src_img)
        #             tgt_image_path = os.path.join(args.data_dir, "item_images", id2image_name[tgt_item_id])
        #             tgt_img = Image.open(tgt_image_path)
        #             tgt_img = tgt_img.convert("RGB")
        #             tgt_input = transform_fn(tgt_img)
        #             self.lines.append((item_label, src_item_id, src_input, tgt_item_id, tgt_input))
        #         except FileNotFoundError:
        #             logger.warning(f"FileNotFoundError: {src_item_id} - {tgt_item_id}")
        #         except Exception as e:
        #             logger.warning(f"Error transforming image: {src_item_id} - {tgt_item_id} ", str(e))
        #         if i % 1000 == 0:
        #             logger.info(f"Finished loading {i} {file_type} data")
        #             # break
        #         i += 1

        for data in datas:
            item_label, src_item_id, src_image_path, tgt_item_id, tgt_image_path = data
            try:
                src_img = Image.open(src_image_path)
                src_input = src_img.convert("RGB")
                # src_input = transform_fn(src_input)
                tgt_img = Image.open(tgt_image_path)
                tgt_input = tgt_img.convert("RGB")
                # tgt_input = transform_fn(tgt_input)
                self.lines.append((item_label, src_item_id, src_input, tgt_item_id, tgt_input))
            except FileNotFoundError:
                logger.warning(f"FileNotFoundError: {src_item_id} - {tgt_item_id}")
            except Exception as e:
                logger.warning(f"Error transforming image: {src_item_id} - {tgt_item_id} ", str(e))
            if i % 1000 == 0:
                logger.info(f"Finished loading {i} {file_type} data")
                # print(f"Finished loading {i} {file_type} data")
                # break
            i += 1

        self.num_lines = len(self.lines)
        if shuffle:
            random.shuffle(self.lines)

        # self.num_file = numfile
        # self.image_feature_file_name = image_feature_file_name
        # self.image_feature_file_name = os.path.join(corpus_path, filetype+'.tsv.%d')
        # print(self.name)
        # if given_file_id:
        #     self.infiles = [self.name % i for i in given_file_id]
        # else:#没给就是所有的
        #     self.infiles = [self.name % i for i in range(self.num_file)]
        # for index, the_file in enumerate(self.infiles):
        #     print(index,':',the_file)#文件排个序
        # self.counts = []
        # self.num_caps = num_caps#
        # if file_type == 'train':
        #     all_df = pd.read_csv('./data/image_lmdb_json/df_train.csv', encoding='utf-8',
        #                          dtype={'image_id': str, 'item_ID': str})#指定类型
        # else:
        #     all_df = pd.read_csv('./data/image_lmdb_json/df_val.csv', encoding='utf-8',
        #                          dtype={'image_id': str, 'item_ID': str})
        #
        # for image_id, pv, caption, category in zip(all_df['image_id'], all_df['pv'], all_df['caption'], all_df['category']):
        #     self.cap_pv_cls[image_id] = (pv, caption, category)

    def __len__(self):
        return self.num_lines

    def __iter__(self):
        for line in self.lines:
            yield line


def image_data(args):
    serializer = td.LMDBSerializer

    train_data, valid_data, test_data = load_raw_data(args)

    # save train data
    if "train" in args.dtypes:
        transform_train_fn = create_transform(input_size=args.image_size, is_training=True,
                                              hflip=args.hflip, color_jitter=args.color_jitter)
        # ds = ImageData(args, train_data, "train", transform_train_fn, id2image_name=None, shuffle=True)
        # # ds = td.PrefetchDataZMQ(ds, 1)
        # out_file = os.path.join(args.output_dir, f"train_feat.lmdb")
        # if os.path.isfile(out_file):
        #     os.remove(out_file)
        # logger.info(f"train data length: {len(ds)}")
        # print(f"train data length: {len(ds)}")
        # try:
        #     serializer.save(ds, out_file)
        # except Exception as e:
        #     logger.error(f"[Error] serialization of train data", e)
        #     # traceback.print_exc()

        ct_batch = len(train_data) // args.batch_size + 1
        for i in range(ct_batch):
            if i < ct_batch - 1:
                datas = train_data[i*args.batch_size:(i+1)*args.batch_size]
            else:
                datas = train_data[i*args.batch_size:]
            ds = ImageData(args, datas, "train", transform_train_fn, id2image_name=None, shuffle=True)
            # ds = td.PrefetchDataZMQ(ds, 1)
            out_file = os.path.join(args.output_dir, f"train_feat_{i+1}.lmdb")
            if os.path.isfile(out_file):
                os.remove(out_file)
            logger.info(f"{i}-th train data length: {len(ds)}")
            # print(f"{i}-th train data length: {len(ds)}")
            try:
                serializer.save(ds, out_file)
            except Exception as e:
                logger.error(f"[Error] serialization of {i}-th train data", e)
                # traceback.print_exc()

    # save eval data
    if "valid" in args.dtypes:
        transform_eval_fn = create_transform(input_size=args.image_size, is_training=False,
                                              hflip=args.hflip, color_jitter=args.color_jitter)
        # ds = ImageData(args, valid_data, "valid", transform_eval_fn, id2image_name=None, shuffle=False)
        # # ds = td.PrefetchDataZMQ(ds, 1)
        # out_file = os.path.join(args.output_dir, f"valid_feat.lmdb")
        # if os.path.isfile(out_file):
        #     os.remove(out_file)
        # logger.info(f"valid data length: {len(ds)}")
        # print(f"valid data length: {len(ds)}")
        # try:
        #     serializer.save(ds, out_file)
        # except Exception as e:
        #     logger.error(f"[Error] serialization of valid data", e)
        #     # traceback.print_exc()

        ct_batch = len(valid_data) // args.batch_size + 1
        for i in range(ct_batch):
            if i < ct_batch - 1:
                datas = valid_data[i*args.batch_size:(i+1)*args.batch_size]
            else:
                datas = valid_data[i*args.batch_size:]
            ds = ImageData(args, datas, "valid", transform_eval_fn, id2image_name=None, shuffle=False)
            # ds = td.PrefetchDataZMQ(ds, 1)
            out_file = os.path.join(args.output_dir, f"valid_feat_{i+1}.lmdb")
            if os.path.isfile(out_file):
                os.remove(out_file)
            logger.info(f"{i}-th valid data length: {len(ds)}")
            # print(f"{i}-th valid data length: {len(ds)}")
            try:
                serializer.save(ds, out_file)
            except Exception as e:
                logger.error(f"[Error] serialization of {i}-th valid data", e)
                # traceback.print_exc()

    # save test data
    if "test" in args.dtypes:
        transform_test_fn = create_transform(input_size=args.image_size, is_training=False,
                                              hflip=args.hflip, color_jitter=args.color_jitter)
        ct_batch = len(test_data) // args.batch_size + 1
        for i in range(ct_batch):
            if i < ct_batch - 1:
                datas = test_data[i*args.batch_size:(i+1)*args.batch_size]
            else:
                datas = test_data[i*args.batch_size:]
            ds = ImageData(args, datas, "test", transform_test_fn, id2image_name=None, shuffle=False)
            ds = td.PrefetchDataZMQ(ds, 1)
            out_file = os.path.join(args.output_dir, f"test_feat_{i+1}.lmdb")
            if os.path.isfile(out_file):
                os.remove(out_file)
            logger.info(f"{i}-th test data length: {len(ds)}")
            try:
                serializer.save(ds, out_file)
            except Exception as e:
                logger.error(f"[Error] serialization of {i}-th test data", e)
                # traceback.print_exc()


def object_detection(args):
    from src.utils import save_one_box
    # load model
    model = torch.hub.load(args.code_path, args.cv_model_name, model_path=args.pretrained_model_path,
                           pretrained=True, source='local')

    # load raw image data
    id2image_name = dict()
    with open(os.path.join(args.data_dir, "item_info.jsonl"), "r", encoding="utf-8") as r:
        while True:
            line = r.readline()
            if not line:
                break
            d = json.loads(line.strip())
            id2image_name[d['item_id']] = d
    logger.info(f"Finished load item info, size: {len(id2image_name)}")

    # Object detection
    for i, (item_id, d) in enumerate(id2image_name.items()):
        file = os.path.join(args.output_dir, "item_images_cropped", f"{item_id}.jpg")
        cate_name = d['cate_name']
        default_cls = CATE2YOLO_CLASS.get(cate_name, None)
        image_path = os.path.join(args.data_dir, "item_images", d['item_image_name'])
        try:
            if default_cls is not None:
                results = model(image_path)
                im = results.imgs[0]
                h, w, _ = im.shape
                preds = results.pred[0].cpu().detach().numpy()
                preds = sorted(preds, key=lambda box: np.abs(box[3] - box[1]) * np.abs(box[2] - box[0]), reverse=True)
                for *box, prob, idx in preds:
                    cls = results.names[int(idx)]
                    if cls in default_cls:
                        box_h = np.abs(box[3] - box[1])
                        box_w = np.abs(box[2] - box[0])
                        ratio = (box_h*box_w) / (h*w)
                        # 裁剪后的图片占原图比例要超过一定阈值才保存
                        if ratio > args.min_crop_ratio:
                            save_one_box(box, im, file=Path(file), save=True)
                            break
                # 若未找到该品类对应的yolov5 class，则拷贝原图
                else:
                    os.system(f"cp {image_path} {file}")
            # 若该品类无对应的yolov5 class，也拷贝原图
            else:
                os.system(f"cp {image_path} {file}")
        except FileNotFoundError:
            logger.warning(f"{item_id} image not found")
        except Exception as e:
            logger.warning(f"Error in detecting {item_id}", e)
            traceback.print_exc()

        if i % 10000 == 0:
            logger.info(f"{i} images cropped")

    logger.info(f"Finished processing all images!")


def main():
    args = get_parser()

    if args.only_image:
        if args.object_detection:
            object_detection(args)
        else:
            image_data(args)
    else:
        if args.with_image:
            img_emb_dict = load_image_embedding(args)
        else:
            img_emb_dict = None

        item_dict, relation_count = pkgm_pretraining_data(args)

        finetune_data(args, item_dict, relation_count, img_emb_dict)


if __name__ == "__main__":
    main()
