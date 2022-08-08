
import os
import json
import argparse
import copy

from src.utils import logger

# 只在验证集出现，未在训练集出现的品类
ONLY_VALID_CATES = ['投资贵金属', '客厅吸顶灯', '衬衫', '电热水壶', '养生壶/煎药壶', '鞋柜', '脱毛膏', '自热火锅', '洗烘套装', '椰棕床垫', '足浴器', '茶壶', '电动自行车']
# 只在测试集出现，未在训练集出现的品类
ONLY_TEST_CATES = ['鞋柜', '洗衣机', '衬衫', '茶壶', '电动自行车', '脱毛膏', '投资贵金属', '椰棕床垫', '身体乳液', '客厅吸顶灯', '电热水壶', '足浴器', '养生壶/煎药壶', '洗烘套装', '自热火锅']

# 模型和阈值
models_and_thresholds = [
    #     ("roberta_base-v3.4-one_tower-cls-ce", 0.4),
    ("roberta_large-v3.4-one_tower-cls-NA-ce", 0.3, 0.8610),
    #     ("roberta_large-v3.6-one_tower-cls-NA-ce", 0.4, 0.8478),
    ("roberta_large-v3.4-one_tower-cls_1,2,3,4_cat-NA-ce", 0.4, 0.8600),
    ("roberta_large-v4-one_tower-cls-NA-ce", 0.4, 0.8612),
    ("roberta_image_large-v5-one_tower-cls-begin-ce", 0.4, 0.8582),
    #     ("roberta_image_large-v5.1-one_tower-cls-begin-ce", 0.4, 0.8446),
    ("eca_nfnet_l0-v6", 0.5, 0.7777),
    ("pkgm_large-v3.4-one_tower-cls-NA-ce", 0.4, 0.8096),
    #     ("item_alignment-k3m_base", 0.6, 0.7635),
    ("bert_base-one_tower-cls-NA-ce", 0.3, 0.8510),
    #     ("bert_adversarial-two_tower-cls-ce", 0.3, 0.8477),
    #     ("fasttext", 0.5, 0.7024),
    ("textcnn-v3.4-two_tower-cls-NA-ce", 0.6, 0.7703),
    #     ("coca_base-v5.2-two_tower-cls-sum-ce", 0.5, 0.7875),
    #     ("coca_large-v5.2-two_tower-cls-sum-ce", 0.5, 0.7784),
    #     ("vit_base_patch16_384-v6.2", 0.5, 0.7685)
]

models_and_thresholds_in = [
    #     ("roberta_base-v3.4-one_tower-cls-ce", 0.4),
    ("roberta_large-v3.4-one_tower-cls-NA-ce", 0.3, 0.8610),
    #     ("roberta_large-v3.6-one_tower-cls-NA-ce", 0.4, 0.8478),
    ("roberta_large-v3.4-one_tower-cls_1,2,3,4_cat-NA-ce", 0.4, 0.8600),
    ("roberta_large-v4-one_tower-cls-NA-ce", 0.3, 0.8612),
    ("roberta_image_large-v5-one_tower-cls-begin-ce", 0.4, 0.8582),
    #     ("roberta_image_large-v5.1-one_tower-cls-begin-ce", 0.4, 0.8446),
    ("eca_nfnet_l0-v6", 0.4, 0.7777),
    ("pkgm_large-v3.4-one_tower-cls-NA-ce", 0.4, 0.8096),
    #     ("item_alignment-k3m_base", 0.6, 0.7635),
    ("bert_base-one_tower-cls-NA-ce", 0.3, 0.8510),
    #     ("bert_adversarial-two_tower-cls-ce", 0.3, 0.8477),
    #     ("fasttext", 0.5, 0.7024),
    ("textcnn-v3.4-two_tower-cls-NA-ce", 0.6, 0.7703),
    #     ("coca_base-v5.2-two_tower-cls-sum-ce", 0.5, 0.7875),
    #     ("coca_large-v5.2-two_tower-cls-sum-ce", 0.5, 0.7784),
    #     ("vit_base_patch16_384-v6.2", 0.5, 0.7685)
]

models_and_thresholds_not_in = [
    ("roberta_large-v3.4-one_tower-cls-NA-ce", 0.4, 0.8610),
    #     ("roberta_large-v3.6-one_tower-cls-NA-ce", 0.4, 0.8583),
    ("roberta_large-v3.4-one_tower-cls_1,2,3,4_cat-NA-ce", 0.4, 0.8600),
    ("roberta_large-v4-one_tower-cls-NA-ce", 0.5, 0.8612),
    ("roberta_image_large-v5-one_tower-cls-begin-ce", 0.4, 0.8582),
    #     ("roberta_image_large-v5.1-one_tower-cls-begin-ce", 0.4, 0.8446),
    #     ("eca_nfnet_l0-v6", 0.5, 0.7783),
    ("pkgm_large-v3.4-one_tower-cls-NA-ce", 0.5, 0.8096),
    #     ("item_alignment-k3m_base", 0.6, 0.7635),
    ("bert_base-one_tower-cls-NA-ce", 0.4, 0.8510),
    #     ("bert_adversarial-two_tower-cls-ce", 0.3, 0.8477),
    #     ("fasttext", 0.5, 0.7024),
    ("textcnn-v3.4-two_tower-cls-NA-ce", 0.6, 0.7703),
    #     ("coca_base-v5.2-two_tower-cls-sum-ce", 0.5, 0.7875),
    #     ("coca_large-v5.2-two_tower-cls-sum-ce", 0.5, 0.7882),
    #     ("vit_base_patch16_384-v6.2", 0.5, 0.7685)
]


def get_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", required=True, type=str, help="数据地址")
    parser.add_argument("--ensemble_strategy", required=True, type=str, help="ensemble strategy: threshold, f1")

    parser.add_argument("--input_file", default="deepAI_result_threshold=0.4.jsonl", type=str,
                        help="input file name")
    parser.add_argument("--split_by_valid_or_test", action="store_true", help="whether to use different models and thresholds based"
                                                                              "on whether the categories appeared in training data")

    return parser.parse_args()


def ensemble(args, id_dict):
    if args.split_by_valid_or_test:
        # 处理在训练集里出现过的品类
        lines = dict()
        for model, threshold, f1 in models_and_thresholds_in:
            f = os.path.join(args.data_dir, "output", model, args.input_file)
            ct = 0
            total = 0
            with open(f, "r", encoding="utf-8") as r:
                while True:
                    line = r.readline()
                    if not line:
                        break
                    d = json.loads(line.strip())
                    src_item_id = d['src_item_id']
                    src_cate_name = id_dict[src_item_id]['cate_name']
                    tgt_item_id = d['tgt_item_id']
                    tgt_cate_name = id_dict[tgt_item_id]['cate_name']
                    # if src_cate_name in ONLY_VALID_CATES or tgt_cate_name in ONLY_VALID_CATES:
                    if src_cate_name in ONLY_TEST_CATES or tgt_cate_name in ONLY_TEST_CATES:
                        continue
                    key = src_item_id + "-" + tgt_item_id
                    prob = eval(d['tgt_item_emb'])[0]
                    if key not in lines:
                        dd = copy.deepcopy(d)
                        dd['tgt_item_emb'] = prob - threshold
                        dd['0'] = 0.0
                        dd['1'] = 0.0
                        lines[key] = dd
                    else:
                        lines[key]['tgt_item_emb'] += prob - threshold
                    if prob >= threshold:
                        ct += 1
                        lines[key]['1'] += f1
                    else:
                        lines[key]['0'] += f1
                    total += 1
            logger.info(f"In Train: {model}-{threshold} p: {ct}, total: {total}")

        # 处理未在训练集里出现过的品类
        for model, threshold, f1 in models_and_thresholds_not_in:
            f = os.path.join(args.data_dir, "output", model, args.input_file)
            ct = 0
            total = 0
            with open(f, "r", encoding="utf-8") as r:
                while True:
                    line = r.readline()
                    if not line:
                        break
                    d = json.loads(line.strip())
                    src_item_id = d['src_item_id']
                    src_cate_name = id_dict[src_item_id]['cate_name']
                    tgt_item_id = d['tgt_item_id']
                    tgt_cate_name = id_dict[tgt_item_id]['cate_name']
                    # if src_cate_name in ONLY_VALID_CATES or tgt_cate_name in ONLY_VALID_CATES:
                    if src_cate_name in ONLY_TEST_CATES or tgt_cate_name in ONLY_TEST_CATES:
                        key = src_item_id + "-" + tgt_item_id
                        prob = eval(d['tgt_item_emb'])[0]
                        if key not in lines:
                            dd = copy.deepcopy(d)
                            dd['tgt_item_emb'] = prob - threshold
                            dd['0'] = 0.0
                            dd['1'] = 0.0
                            lines[key] = dd
                        else:
                            lines[key]['tgt_item_emb'] += prob - threshold
                        if prob >= threshold:
                            ct += 1
                            lines[key]['1'] += f1
                        else:
                            lines[key]['0'] += f1
                        total += 1
            logger.info(f"Not In Train: {model}-{threshold} p: {ct}, total: {total}")
    else:
        lines = dict()
        for model, threshold, f1 in models_and_thresholds:
            f = os.path.join(args.data_dir, "output", model, args.input_file)
            ct = 0
            total = 0
            with open(f, "r", encoding="utf-8") as r:
                while True:
                    line = r.readline()
                    if not line:
                        break
                    d = json.loads(line.strip())
                    src_item_id = d['src_item_id']
                    src_cate_name = id_dict[src_item_id]['cate_name']
                    tgt_item_id = d['tgt_item_id']
                    tgt_cate_name = id_dict[tgt_item_id]['cate_name']
                    key = src_item_id + "-" + tgt_item_id
                    prob = eval(d['tgt_item_emb'])[0]
                    if key not in lines:
                        dd = copy.deepcopy(d)
                        dd['tgt_item_emb'] = prob - threshold
                        dd['0'] = 0.0
                        dd['1'] = 0.0
                        lines[key] = dd
                    else:
                        lines[key]['tgt_item_emb'] += prob - threshold
                    if prob >= threshold:
                        ct += 1
                        lines[key]['1'] += f1
                    else:
                        lines[key]['0'] += f1
                    total += 1
            logger.info(f"{model}-{threshold} p: {ct}, total: {total}")

    return lines


def main():
    args = get_parser()

    # 加载item信息文件
    id_dict = dict()
    with open(os.path.join(args.data_dir, "raw", "item_info.jsonl"), "r", encoding="utf-8") as r:
        while True:
            line = r.readline()
            if not line:
                break
            d = json.loads(line.strip())
            item_id = d['item_id']
            id_dict[item_id] = d
    logger.info(f"id dict length: {len(id_dict)}")

    # 加载各个模型的结果
    lines = ensemble(args, id_dict)

    # 模型结果融合
    lines_ensemble = []
    threshold = 0.0
    total = 0
    ct = 0
    cv_included_ct = 0
    for _, d in lines.items():
        dd = copy.deepcopy(d)
        if args.ensemble_strategy == "f1":
            if dd['1'] >= dd['0']:
                ct += 1
                p = 1.0
            else:
                p = -1.0
        elif args.ensemble_strategy == "threshold":
            if dd['tgt_item_emb'] >= threshold:
                ct += 1
            p = dd['tgt_item_emb']
        else:
            raise ValueError(f"unsupported ensemble strategy: {args.ensemble_strategy}")
        dd['tgt_item_emb'] = f"[{p}]"
        dd['threshold'] = threshold
        total += 1
        lines_ensemble.append(dd)
    logger.info(f"cv included p: {cv_included_ct}, p: {ct}, total: {total}")

    # 模型结果保存
    # model = "ensemble_f1-rl_v3.4_0.3-rl_v3.6_0.4-ril_v5_0.4-el0_v6_0.4-cl_v5.2_0.5-vb_v6_0.4"
    # model = "ensemble-rl_v3.4_0.3-rlcat_v3.4_0.4-rl_v4_0.4-ril_v5_0.4-el0_v6_0.5-pl_v3.4_0.4-bb_0.3-tc_v3.4_0.6-cl_v5.2_0.5"
    model = "ensemble"
    model_dir = os.path.join(args.data_dir, "output", model)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    f = os.path.join(model_dir, f"deepAI_result.jsonl")
    with open(f, "w", encoding="utf-8") as w:
        for dd in lines_ensemble:
            w.write(json.dumps(dd)+"\n")


if __name__ == "__main__":
    main()
