# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :commodity-alignment
# @File     :convert_data
# @Date     :2022/8/8 11:38
# @Author   :mengqingyang
# @Email    :mengqingyang0102@163.com
-------------------------------------------------
"""
import json
from typing import List
import math


def compute(item_emb_1:List[float], item_emb_2:List[float]) -> float:
    s = 0.0
    for a, b in zip(item_emb_1, item_emb_2):
        s += a * b
    return s


def convert_data_format(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as reader,\
            open(output_file, "w", encoding="utf-8") as writer:
        for line in reader:
            record = json.loads(line.strip())
            src_emb = [float(num) for num in record["src_item_emb"].strip('[').strip(']').split(',')]
            tgt_emb = [float(num) for num in record["tgt_item_emb"].strip('[').strip(']').split(',')]
            threshold = float(record['threshold'])
            prob = compute(src_emb, tgt_emb)

            prob = 1 / (1 + math.exp(-prob + threshold))
            rec = {
                "src_item_id": record["src_item_id"],
                "tgt_item_id": record["tgt_item_id"],
                "src_item_emb": str([1 - prob]),
                "tgt_item_emb": str([prob]),
                "threshold": 0.5
            }
            writer.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    input_file = ""
    output_file = ""
    convert_data_format(input_file, output_file)


