import jieba
import torch

from torch.utils.data import Dataset
from PIL import Image
from timm.data.transforms_factory import create_transform
from transformers.tokenization_utils import TruncationStrategy, PaddingStrategy

IMG_TOKEN = "[unused99]"
IMG_TOKEN_ID = 99
COLON_ID = 131
SEMICOLON_ID = 132


def collate_coca(inputs):
    item_ids, input_ids, attention_mask, token_type_ids, position_ids, images = [], [], [], [], [], []
    for inp in inputs:
        image = inp.get('image', None)
        if image is not None:
            images.append(image)
            input_ids.append(inp["input_ids"])
            attention_mask.append(inp["attention_mask"])
            token_type_ids.append(inp["token_type_ids"])
            if "position_ids" in inp:
                position_ids.append(inp["position_ids"])
            item_ids.append(inp['item_id'])

    images = torch.stack(images)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    position_ids = torch.tensor(position_ids, dtype=torch.long) if len(position_ids) > 0 else None

    return item_ids, input_ids, attention_mask, token_type_ids, position_ids, images


def collate_coca_pair(inputs):
    src_item_ids, src_input_ids, src_attention_mask, src_token_type_ids, src_position_ids, src_images = [], [], [], [], [], []
    tgt_item_ids, tgt_input_ids, tgt_attention_mask, tgt_token_type_ids, tgt_position_ids, tgt_images = [], [], [], [], [], []
    labels = []
    for inp in inputs:
        src_image = inp.get('src_image', None)
        tgt_image = inp.get('tgt_image', None)
        if src_image is not None and tgt_image is not None:
            labels.append(inp["labels"])
            src_images.append(src_image)
            src_input_ids.append(inp["src_input_ids"])
            src_attention_mask.append(inp["src_attention_mask"])
            src_token_type_ids.append(inp["src_token_type_ids"])
            if "src_position_ids" in inp:
                src_position_ids.append(inp["src_position_ids"])
            src_item_ids.append(inp['src_item_id'])
            tgt_images.append(tgt_image)
            tgt_input_ids.append(inp["tgt_input_ids"])
            tgt_attention_mask.append(inp["tgt_attention_mask"])
            tgt_token_type_ids.append(inp["tgt_token_type_ids"])
            if "tgt_position_ids" in inp:
                tgt_position_ids.append(inp["tgt_position_ids"])
            tgt_item_ids.append(inp['tgt_item_id'])

    labels = torch.tensor(labels, dtype=torch.long)
    src_images = torch.stack(src_images)
    src_input_ids = torch.tensor(src_input_ids, dtype=torch.long)
    src_attention_mask = torch.tensor(src_attention_mask, dtype=torch.long)
    src_token_type_ids = torch.tensor(src_token_type_ids, dtype=torch.long)
    src_position_ids = torch.tensor(src_position_ids, dtype=torch.long) if len(src_position_ids) > 0 else None
    tgt_images = torch.stack(tgt_images)
    tgt_input_ids = torch.tensor(tgt_input_ids, dtype=torch.long)
    tgt_attention_mask = torch.tensor(tgt_attention_mask, dtype=torch.long)
    tgt_token_type_ids = torch.tensor(tgt_token_type_ids, dtype=torch.long)
    tgt_position_ids = torch.tensor(tgt_position_ids, dtype=torch.long) if len(tgt_position_ids) > 0 else None

    return src_item_ids, tgt_item_ids, src_input_ids, src_attention_mask, src_token_type_ids, src_position_ids, \
        src_images, tgt_input_ids, tgt_attention_mask, tgt_token_type_ids, tgt_position_ids, tgt_images, labels


def collate_image(inputs):
    src_inputs, tgt_inputs, labels = [], [], []
    src_item_ids = []
    tgt_item_ids = []
    for inp in inputs:
        src_input = inp.get('src_input', None)
        tgt_input = inp.get('tgt_input', None)
        if src_input is not None and tgt_input is not None:
            labels.append(inp["labels"])
            src_item_ids.append(inp['src_item_id'])
            tgt_item_ids.append(inp['tgt_item_id'])
            src_inputs.append(src_input)
            tgt_inputs.append(tgt_input)

    labels = torch.tensor(labels, dtype=torch.long)
    src_inputs = torch.stack(src_inputs)
    tgt_inputs = torch.stack(tgt_inputs)

    return src_item_ids, tgt_item_ids, src_inputs, tgt_inputs, labels


def collate_multimodal(inputs):
    input_ids, attention_mask, token_type_ids, position_ids, labels, image_indices = [], [], [], [], [], []
    src_item_ids = []
    tgt_item_ids = []
    src_img_embs = []
    tgt_img_embs = []
    for inp in inputs:
        input_ids.append(inp["input_ids"])
        attention_mask.append(inp["attention_mask"])
        token_type_ids.append(inp["token_type_ids"])
        if "position_ids" in inp:
            position_ids.append(inp["position_ids"])
        labels.append(inp["labels"])
        src_item_ids.append(inp['src_item_id'])
        tgt_item_ids.append(inp['tgt_item_id'])
        src_img_embs.append(inp['src_img_emb'])
        tgt_img_embs.append(inp['tgt_img_emb'])
        if 'image_index' in inp:
            image_indices.append(inp['image_index'])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    position_ids = torch.tensor(position_ids, dtype=torch.long) if len(position_ids) > 0 else None
    labels = torch.tensor(labels, dtype=torch.long)
    src_img_embs = torch.tensor(src_img_embs, dtype=torch.float32)
    tgt_img_embs = torch.tensor(tgt_img_embs, dtype=torch.float32)
    image_indices = torch.tensor(image_indices, dtype=torch.long)

    return src_item_ids, tgt_item_ids, image_indices, src_img_embs, tgt_img_embs, \
           input_ids, token_type_ids, attention_mask, position_ids, labels


def collate_multimodal_two_tower(inputs):
    src_item_ids, src_img_embs, input_ids_1, attention_mask_1, token_type_ids_1 = [], [], [], [], []
    tgt_item_ids, tgt_img_embs, input_ids_2, attention_mask_2, token_type_ids_2 = [], [], [], [], []
    # image_indices = []
    labels = []
    position_ids = []
    for inp in inputs:
        input_ids_1.append(inp["input_ids_1"])
        attention_mask_1.append(inp["attention_mask_1"])
        token_type_ids_1.append(inp["token_type_ids_1"])
        input_ids_2.append(inp["input_ids_2"])
        attention_mask_2.append(inp["attention_mask_2"])
        token_type_ids_2.append(inp["token_type_ids_2"])
        if "position_ids" in inp:
            position_ids.append(inp["position_ids"])
        labels.append(inp["labels"])
        src_item_ids.append(inp['src_item_id'])
        tgt_item_ids.append(inp['tgt_item_id'])
        src_img_embs.append(inp['src_img_emb'])
        tgt_img_embs.append(inp['tgt_img_emb'])
        # if 'image_index' in inp:
        #     image_indices.append(inp['image_index'])

    input_ids_1 = torch.tensor(input_ids_1, dtype=torch.long)
    attention_mask_1 = torch.tensor(attention_mask_1, dtype=torch.long)
    token_type_ids_1 = torch.tensor(token_type_ids_1, dtype=torch.long)
    # cate_ids_1 = torch.tensor(cate_ids_1, dtype=torch.long)
    input_ids_2 = torch.tensor(input_ids_2, dtype=torch.long)
    attention_mask_2 = torch.tensor(attention_mask_2, dtype=torch.long)
    token_type_ids_2 = torch.tensor(token_type_ids_2, dtype=torch.long)
    # cate_ids_2 = torch.tensor(cate_ids_2, dtype=torch.long)
    position_ids = torch.tensor(position_ids, dtype=torch.long) if len(position_ids) > 0 else None
    labels = torch.tensor(labels, dtype=torch.long)
    src_img_embs = torch.tensor(src_img_embs, dtype=torch.float32)
    tgt_img_embs = torch.tensor(tgt_img_embs, dtype=torch.float32)
    # image_indices = torch.tensor(image_indices, dtype=torch.long)

    return src_item_ids, tgt_item_ids, input_ids_1, attention_mask_1, token_type_ids_1, position_ids, src_img_embs, \
           input_ids_2, attention_mask_2, token_type_ids_2, position_ids, tgt_img_embs, labels


def collate_one_tower(inputs):
    input_ids, attention_mask, token_type_ids, position_ids, labels = [], [], [], [], []
    pair_indices = []
    src_item_ids = []
    tgt_item_ids = []
    for inp in inputs:
        input_ids.append(inp["input_ids"])
        attention_mask.append(inp["attention_mask"])
        token_type_ids.append(inp["token_type_ids"])
        if "position_ids" in inp:
            position_ids.append(inp["position_ids"])
        if "pair_indices" in inp:
            pair_indices.append(inp['pair_indices'])
        # if "cate_ids" in inp:
        #     cate_ids.append(inp["cate_ids"])
        labels.append(inp["labels"])
        src_item_ids.append(inp['src_item_id'])
        tgt_item_ids.append(inp['tgt_item_id'])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    position_ids = torch.tensor(position_ids, dtype=torch.long) if len(position_ids) > 0 else None
    # cate_ids = torch.tensor(cate_ids, dtype=torch.long) if len(cate_ids) > 0 else None
    labels = torch.tensor(labels, dtype=torch.long)
    # pair_indices = torch.tensor(pair_indices, dtype=torch.long)
    pair_indices = [torch.tensor(pair_index, dtype=torch.long) for pair_index in pair_indices]

    return src_item_ids, tgt_item_ids, pair_indices, input_ids, \
           token_type_ids, attention_mask, position_ids, labels


def collate_two_tower(inputs):
    input_ids_1, attention_mask_1, token_type_ids_1, cate_ids_1 = [], [], [], []
    input_ids_2, attention_mask_2, token_type_ids_2, cate_ids_2 = [], [], [], []
    position_ids = []
    src_item_ids = []
    tgt_item_ids = []
    labels = []
    for inp in inputs:
        input_ids_1.append(inp["input_ids_1"])
        attention_mask_1.append(inp["attention_mask_1"])
        token_type_ids_1.append(inp["token_type_ids_1"])
        input_ids_2.append(inp["input_ids_2"])
        attention_mask_2.append(inp["attention_mask_2"])
        token_type_ids_2.append(inp["token_type_ids_2"])
        if "position_ids" in inp:
            position_ids.append(inp["position_ids"])
        if "cate_ids_1" in inp:
            cate_ids_1.append(inp['cate_ids_1'])
        if "cate_ids_2" in inp:
            cate_ids_2.append(inp['cate_ids_2'])
        labels.append(inp["labels"])
        src_item_ids.append(inp['src_item_id'])
        tgt_item_ids.append(inp['tgt_item_id'])

    input_ids_1 = torch.tensor(input_ids_1, dtype=torch.long)
    attention_mask_1 = torch.tensor(attention_mask_1, dtype=torch.long)
    token_type_ids_1 = torch.tensor(token_type_ids_1, dtype=torch.long)
    # cate_ids_1 = torch.tensor(cate_ids_1, dtype=torch.long)
    input_ids_2 = torch.tensor(input_ids_2, dtype=torch.long)
    attention_mask_2 = torch.tensor(attention_mask_2, dtype=torch.long)
    token_type_ids_2 = torch.tensor(token_type_ids_2, dtype=torch.long)
    # cate_ids_2 = torch.tensor(cate_ids_2, dtype=torch.long)
    position_ids = torch.tensor(position_ids, dtype=torch.long) if len(position_ids) > 0 else None
    labels = torch.tensor(labels, dtype=torch.long)

    return src_item_ids, tgt_item_ids, input_ids_1, attention_mask_1, token_type_ids_1, \
           input_ids_2, attention_mask_2, token_type_ids_2, position_ids, labels


def collate(inputs):
    input_ids, attention_mask, token_type_ids, position_ids = [], [], [], []
    ids = []
    for inp in inputs:
        input_ids.append(inp["input_ids"])
        attention_mask.append(inp["attention_mask"])
        token_type_ids.append(inp["token_type_ids"])
        if "position_ids" in inp:
            position_ids.append(inp["position_ids"])
        ids.append(inp['idx'])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    position_ids = torch.tensor(position_ids, dtype=torch.long) if len(position_ids) > 0 else None

    return ids, input_ids, token_type_ids, attention_mask, position_ids


def collate_gnn(inputs):
    # labels = []
    # src_ids, src_item_ids, tgt_ids, tgt_item_ids = [], [], [], []
    # for inp in inputs:
    #     src_ids.append(inp["src_idx"])
    #     src_item_ids.append(inp["src_item_id"])
    #     tgt_ids.append(inp["tgt_idx"])
    #     tgt_item_ids.append(inp["tgt_item_id"])
    #     if "item_label" in inp:
    #         labels.append(inp["item_label"])

    # return labels, src_ids, src_item_ids, tgt_ids, tgt_item_ids
    return inputs


class PKGMOneTowerDataset(Dataset):
    def __init__(self, data, text_tokenizer, kg_entity_tokenizer,
                 kg_relation_tokenizer, max_seq_en, max_pvs,
                 classification_method):
        self.data = data
        self.text_tokenizer = text_tokenizer
        self.kg_entity_tokenizer = kg_entity_tokenizer
        self.kg_relation_tokenizer = kg_relation_tokenizer
        self.max_seq_len = max_seq_en
        self.max_pvs = max_pvs
        self.classification_method = classification_method

    def __getitem__(self, item):
        record = dict()
        item_label, src_item_id, src_cater_id, src_title, src_pvs, \
            tgt_item_id, tgt_cate_id, tgt_title, tgt_pvs = self.data[item]
        # text tokeinzation
        src_title_tokens = self.text_tokenizer.tokenize(src_title)
        src_title_input_ids = self.text_tokenizer.convert_tokens_to_ids(src_title_tokens)
        tgt_title_tokens = self.text_tokenizer.tokenize(tgt_title)
        tgt_title_input_ids = self.text_tokenizer.convert_tokens_to_ids(tgt_title_tokens)

        # kg tokenization
        src_pv_input_ids = []
        for pv in src_pvs.split(";"):
            try:
                r, t = pv.split(":", maxsplit=1)
            except ValueError:
                continue
            src_pv_input_ids.append(self.kg_relation_tokenizer[r])
        if len(src_pv_input_ids) > 0:
            src_h = f"/item/{src_item_id}"
            src_pv_input_ids.insert(0, self.kg_entity_tokenizer[src_h])
        tgt_pv_input_ids = []
        for pv in tgt_pvs.split(";"):
            try:
                r, t = pv.split(":", maxsplit=1)
            except ValueError:
                continue
            tgt_pv_input_ids.append(self.kg_relation_tokenizer[r])
        if len(tgt_pv_input_ids) > 0:
            tgt_h = f"/item/{tgt_item_id}"
            tgt_pv_input_ids.insert(0, self.kg_entity_tokenizer[tgt_h])

        # padding
        src_title_input_ids = src_title_input_ids[:(self.max_seq_len-2)]
        src_title_input_ids = [self.text_tokenizer.cls_token_id] + src_title_input_ids + [self.text_tokenizer.sep_token_id]
        src_title_input_ids, src_title_attention_mask, src_title_token_type_ids = self.pad_text_sequence(src_title_input_ids,
                                                                                              token_type_id=0,
                                                                                              max_len=self.max_seq_len)
        src_pv_input_ids = src_pv_input_ids[:(1+self.max_pvs)]
        src_pv_input_ids, src_pv_attention_mask, src_pv_token_type_ids = self.pad_kg_sequence(src_pv_input_ids,
                                                                                           token_type_id=0,
                                                                                           max_len=self.max_pvs)
        tgt_title_input_ids = tgt_title_input_ids[:(self.max_seq_len-2)]
        tgt_title_input_ids = [self.text_tokenizer.bos_token_id if self.classification_method == "vec_sim" else self.text_tokenizer.sep_token_id] +\
                              tgt_title_input_ids + [self.text_tokenizer.sep_token_id]
        tgt_title_input_ids, tgt_title_attention_mask, tgt_title_token_type_ids = self.pad_text_sequence(tgt_title_input_ids,
                                                                                              token_type_id=1,
                                                                                              max_len=self.max_seq_len)
        tgt_pv_input_ids = tgt_pv_input_ids[:(1+self.max_pvs)]
        tgt_pv_input_ids, tgt_pv_attention_mask, tgt_pv_token_type_ids = self.pad_kg_sequence(tgt_pv_input_ids,
                                                                                           token_type_id=1,
                                                                                           max_len=self.max_pvs)

        input_ids = src_title_input_ids + src_pv_input_ids + tgt_title_input_ids + tgt_pv_input_ids
        attention_mask = src_title_attention_mask + src_pv_attention_mask + tgt_title_attention_mask + tgt_pv_attention_mask
        token_type_ids = src_title_token_type_ids + src_pv_token_type_ids + tgt_title_token_type_ids + tgt_pv_token_type_ids
        position_ids = list(range(0, 2*(self.max_seq_len+2*self.max_pvs)))

        # len_input_ids = len(src_title_input_ids) + len(tgt_title_input_ids) + len(src_pv_input_ids) + len(tgt_pv_input_ids)

        # sanity check
        assert len(src_title_input_ids) == self.max_seq_len
        assert len(tgt_title_input_ids) == self.max_seq_len
        assert len(src_pv_input_ids) == self.max_pvs + 1
        assert len(tgt_pv_input_ids) == self.max_pvs + 1
        assert len(input_ids) == 2*(self.max_seq_len+self.max_pvs+1)
        assert len(token_type_ids) == len(position_ids)
        assert len(token_type_ids) == len(attention_mask)

        record["input_ids"] = input_ids
        record["token_type_ids"] = token_type_ids
        record["position_ids"] = position_ids
        record["attention_mask"] = attention_mask
        record["labels"] = int(item_label)
        record['src_item_id'] = src_item_id
        record['tgt_item_id'] = tgt_item_id

        return record

    def pad_text_sequence(self, tokens, token_type_id, max_len):
        attention_mask = [1] * len(tokens)
        token_type_ids = [token_type_id] * len(tokens)
        while len(tokens) < max_len:
            tokens.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)

        return tokens, attention_mask, token_type_ids

    def pad_kg_sequence(self, tokens, token_type_id, max_len):
        attention_mask = [1] * (len(tokens)-1) * 2
        token_type_ids = [token_type_id] * (len(tokens)-1) * 2
        while len(tokens) - 1 < max_len:
            tokens.append(0)
            attention_mask.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)
            token_type_ids.append(0)

        return tokens, attention_mask, token_type_ids

    def __len__(self):
        return len(self.data)


class PKGMTwoTowerDataset(Dataset):
    def __init__(self, data, text_tokenizer, kg_entity_tokenizer,
                 kg_relation_tokenizer, max_seq_en, max_pvs):
        self.data = data
        self.text_tokenizer = text_tokenizer
        self.kg_entity_tokenizer = kg_entity_tokenizer
        self.kg_relation_tokenizer = kg_relation_tokenizer
        self.max_seq_len = max_seq_en
        self.max_pvs = max_pvs

    def __getitem__(self, item):
        record = dict()
        item_label, src_item_id, src_cate_id, src_title, src_pvs, \
            tgt_item_id, tgt_cate_id, tgt_title, tgt_pvs = self.data[item]

        position_ids = list(range(0, self.max_seq_len + 2*self.max_pvs))

        # src processing
        # text tokeinzation
        src_title_tokens = self.text_tokenizer.tokenize(src_title)
        src_title_input_ids = self.text_tokenizer.convert_tokens_to_ids(src_title_tokens)
        # kg tokenization
        src_pv_input_ids = []
        for pv in src_pvs.split(";"):
            try:
                r, t = pv.split(":", maxsplit=1)
            except ValueError:
                continue
            src_pv_input_ids.append(self.kg_relation_tokenizer[r])
        if len(src_pv_input_ids) > 0:
            src_h = f"/item/{src_item_id}"
            src_pv_input_ids.insert(0, self.kg_entity_tokenizer[src_h])
        # padding
        src_title_input_ids = src_title_input_ids[:(self.max_seq_len-2)]
        src_title_input_ids = [self.text_tokenizer.cls_token_id] + src_title_input_ids + [self.text_tokenizer.sep_token_id]
        src_title_input_ids, src_title_attention_mask, src_title_token_type_ids = self.pad_text_sequence(src_title_input_ids,
                                                                                                         token_type_id=0,
                                                                                                         max_len=self.max_seq_len)
        src_pv_input_ids = src_pv_input_ids[:(1+self.max_pvs)]
        src_pv_input_ids, src_pv_attention_mask, src_pv_token_type_ids = self.pad_kg_sequence(src_pv_input_ids,
                                                                                              token_type_id=1,
                                                                                              max_len=self.max_pvs)
        src_input_ids = src_title_input_ids + src_pv_input_ids
        src_attention_mask = src_title_attention_mask + src_pv_attention_mask
        src_token_type_ids = src_title_token_type_ids + src_pv_token_type_ids
        # sanity check
        assert len(src_title_input_ids) == self.max_seq_len
        assert len(src_pv_input_ids) == self.max_pvs + 1
        assert len(src_input_ids) == self.max_seq_len + self.max_pvs + 1
        assert len(src_token_type_ids) == len(position_ids)
        assert len(src_attention_mask) == len(src_token_type_ids)

        # tgt processing
        tgt_title_tokens = self.text_tokenizer.tokenize(tgt_title)
        tgt_title_input_ids = self.text_tokenizer.convert_tokens_to_ids(tgt_title_tokens)
        tgt_pv_input_ids = []
        for pv in tgt_pvs.split(";"):
            try:
                r, t = pv.split(":", maxsplit=1)
            except ValueError:
                continue
            tgt_pv_input_ids.append(self.kg_relation_tokenizer[r])
        if len(tgt_pv_input_ids) > 0:
            tgt_h = f"/item/{tgt_item_id}"
            tgt_pv_input_ids.insert(0, self.kg_entity_tokenizer[tgt_h])
        tgt_title_input_ids = tgt_title_input_ids[:(self.max_seq_len-2)]
        tgt_title_input_ids = [self.text_tokenizer.cls_token_id] + tgt_title_input_ids + [self.text_tokenizer.sep_token_id]
        tgt_title_input_ids, tgt_title_attention_mask, tgt_title_token_type_ids = self.pad_text_sequence(tgt_title_input_ids,
                                                                                                         token_type_id=0,
                                                                                                         max_len=self.max_seq_len)
        tgt_pv_input_ids = tgt_pv_input_ids[:(1+self.max_pvs)]
        tgt_pv_input_ids, tgt_pv_attention_mask, tgt_pv_token_type_ids = self.pad_kg_sequence(tgt_pv_input_ids,
                                                                                              token_type_id=1,
                                                                                              max_len=self.max_pvs)
        tgt_input_ids = tgt_title_input_ids + tgt_pv_input_ids
        tgt_attention_mask = tgt_title_attention_mask + tgt_pv_attention_mask
        tgt_token_type_ids = tgt_title_token_type_ids + tgt_pv_token_type_ids
        # len_input_ids = len(src_title_input_ids) + len(tgt_title_input_ids) + len(src_pv_input_ids) + len(tgt_pv_input_ids)

        # sanity check
        assert len(tgt_title_input_ids) == self.max_seq_len
        assert len(tgt_pv_input_ids) == self.max_pvs + 1
        assert len(tgt_input_ids) == self.max_seq_len + self.max_pvs + 1
        assert len(tgt_token_type_ids) == len(position_ids)
        assert len(tgt_token_type_ids) == len(tgt_attention_mask)

        record["input_ids_1"] = src_input_ids
        record["token_type_ids_1"] = src_token_type_ids
        record["attention_mask_1"] = src_attention_mask
        record["input_ids_2"] = tgt_input_ids
        record["token_type_ids_2"] = tgt_token_type_ids
        record["attention_mask_2"] = tgt_attention_mask
        record["position_ids"] = position_ids
        record["labels"] = int(item_label)
        record['src_item_id'] = src_item_id
        record['tgt_item_id'] = tgt_item_id

        return record

    def pad_text_sequence(self, tokens, token_type_id, max_len):
        attention_mask = [1] * len(tokens)
        token_type_ids = [token_type_id] * len(tokens)
        while len(tokens) < max_len:
            tokens.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)

        return tokens, attention_mask, token_type_ids

    def pad_kg_sequence(self, tokens, token_type_id, max_len):
        attention_mask = [1] * (len(tokens)-1) * 2
        token_type_ids = [token_type_id] * (len(tokens)-1) * 2
        while len(tokens) - 1 < max_len:
            tokens.append(0)
            attention_mask.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)
            token_type_ids.append(0)

        return tokens, attention_mask, token_type_ids

    def __len__(self):
        return len(self.data)


class RobertaOneTowerDataset(Dataset):
    def __init__(self, data, text_tokenizer, max_seq_len, classification_method, max_seq_len_pv=None,
                 auxiliary_task=False):
        self.data = data
        self.text_tokenizer = text_tokenizer
        self.max_seq_len = max_seq_len
        self.max_seq_len_pv = max_seq_len_pv
        self.classification_method = classification_method
        self.auxiliary_task = auxiliary_task

    def __getitem__(self, item):
        record = dict()
        item_label, src_item_id, src_cate_id, src_title, src_pvs, \
            tgt_item_id, tgt_cate_id, tgt_title, tgt_pvs = self.data[item]

        if self.max_seq_len is None:
            src_text = src_pvs
            tgt_text = tgt_pvs
            max_length = self.max_seq_len_pv
        elif self.max_seq_len_pv is None:
            src_text = src_title
            tgt_text = tgt_title
            max_length = self.max_seq_len
        else:
            src_text = " ".join((src_title, self.text_tokenizer.sep_token, " ".join(jieba.cut(src_pvs))))
            tgt_text = " ".join((tgt_title, self.text_tokenizer.sep_token, " ".join(jieba.cut(tgt_pvs))))
            max_length = self.max_seq_len + self.max_seq_len_pv

        if self.classification_method == "vec_sim":
            src = self.text_tokenizer(text=src_text, max_length=max_length, padding=PaddingStrategy.MAX_LENGTH,
                                      truncation=TruncationStrategy.LONGEST_FIRST).data
            tgt = self.text_tokenizer(text=tgt_text, max_length=max_length, padding=PaddingStrategy.MAX_LENGTH,
                                      truncation=TruncationStrategy.LONGEST_FIRST).data
            record["input_ids"] = src['input_ids'] + [self.text_tokenizer.bos_token_id] + tgt['input_ids'][1:]
            record["token_type_ids"] = src['token_type_ids'] + [ttid+1 for ttid in tgt['token_type_ids']]
            record["attention_mask"] = src['attention_mask'] + tgt['attention_mask']
            len_src = len(src['input_ids'])
            len_tgt = len(tgt['input_ids'])
        else:
            record = self.text_tokenizer(text=src_text, text_pair=tgt_text, max_length=2*max_length,
                                         padding=PaddingStrategy.MAX_LENGTH, truncation=TruncationStrategy.LONGEST_FIRST).data
            pos_sep = [i for i, id_ in enumerate(record['input_ids']) if id_ == self.text_tokenizer.sep_token_id]
            len_src = pos_sep[1]
            len_tgt = len(record['input_ids']) - pos_sep[1]

        record['labels'] = int(item_label)
        record['src_item_id'] = src_item_id
        record['tgt_item_id'] = tgt_item_id
        # record['cate_ids'] = [src_cate_id] * len_src + [tgt_cate_id] * len_tgt
        if self.auxiliary_task:
            pos_sep = [i for i, id_ in enumerate(record['input_ids']) if id_ == self.text_tokenizer.sep_token_id]
            src_pre = pos_sep[0] + 1
            tgt_pre = pos_sep[2] + 1
            src_pvs_ids = record['input_ids'][(pos_sep[0]+1):pos_sep[1]]
            tgt_pvs_ids = record['input_ids'][(pos_sep[2]+1):pos_sep[3]]
            pair_indices = []
            src_p, tgt_p = 0, 0
            src_colon, src_semicolon, src_prev_semicolon = None, -1, None
            tgt_colon, tgt_semicolon, tgt_prev_semicolon = None, -1, None
            while src_p < len(src_pvs_ids) and tgt_p < len(tgt_pvs_ids):
                # src pvs: find colon(:) and semicolon(;) position
                while src_p < len(src_pvs_ids):
                    if src_pvs_ids[src_p] == COLON_ID:
                        src_colon = src_p
                    elif src_pvs_ids[src_p] == SEMICOLON_ID:
                        src_prev_semicolon = src_semicolon
                        src_semicolon = src_p
                        src_p += 1
                        break
                    src_p += 1
                else:
                    break
                # tgt pvs: find colon(:) and semicolon(;) position
                while tgt_p < len(tgt_pvs_ids):
                    if tgt_pvs_ids[tgt_p] == COLON_ID:
                        tgt_colon = tgt_p
                    elif tgt_pvs_ids[tgt_p] == SEMICOLON_ID:
                        tgt_prev_semicolon = tgt_semicolon
                        tgt_semicolon = tgt_p
                        tgt_p += 1
                        break
                    tgt_p += 1
                else:
                    break
                # select key and values in pv
                src_key = src_pvs_ids[(src_prev_semicolon+1):src_colon]
                src_value = src_pvs_ids[(src_colon+1):src_semicolon]
                tgt_key = tgt_pvs_ids[(tgt_prev_semicolon+1):tgt_colon]
                tgt_value = tgt_pvs_ids[(tgt_colon+1):tgt_semicolon]
                # if key is not the same for src and tgt, end
                if src_key != tgt_key:
                    break
                # record: (1) src pv pair position (2) tgt pv pair postion, (3) whether value is the same
                pair_indices.append([src_prev_semicolon+1+src_pre, src_semicolon+src_pre,
                                     tgt_prev_semicolon+1+tgt_pre, tgt_semicolon+tgt_pre,
                                     1 if src_value == tgt_value else 0])
            record['pair_indices'] = pair_indices

        return record

    def __len__(self):
        return len(self.data)


class RobertaImageOneTowerDataset(Dataset):
    def __init__(self, data, text_tokenizer, max_seq_len, ensemble, max_seq_len_pv=None):
        self.data = data
        self.text_tokenizer = text_tokenizer
        self.max_seq_len = max_seq_len
        self.max_seq_len_pv = max_seq_len_pv
        # self.classification_method = classification_method
        self.ensemble = ensemble

    def __getitem__(self, item):
        record = dict()
        item_label, src_item_id, src_title, src_pvs, src_img_emb, \
            tgt_item_id, tgt_title, tgt_pvs, tgt_img_emb = self.data[item]

        if self.max_seq_len is None:
            src_text = src_pvs
            tgt_text = tgt_pvs
            max_length = self.max_seq_len_pv
        elif self.max_seq_len_pv is None:
            src_text = src_title
            tgt_text = tgt_title
            max_length = self.max_seq_len
        else:
            src_text = " ".join((src_title, self.text_tokenizer.sep_token, " ".join(jieba.cut(src_pvs))))
            tgt_text = " ".join((tgt_title, self.text_tokenizer.sep_token, " ".join(jieba.cut(tgt_pvs))))
            max_length = self.max_seq_len + self.max_seq_len_pv

        if self.ensemble == "begin":
            src_text = " ".join((IMG_TOKEN, self.text_tokenizer.sep_token, src_text))
            tgt_text = " ".join((IMG_TOKEN, self.text_tokenizer.sep_token, tgt_text))

        # if self.classification_method == "vec_sim":
        #     src = self.text_tokenizer(text=src_text, max_length=max_length, padding=PaddingStrategy.MAX_LENGTH,
        #                               truncation=TruncationStrategy.LONGEST_FIRST).data
        #     tgt = self.text_tokenizer(text=tgt_text, max_length=max_length, padding=PaddingStrategy.MAX_LENGTH,
        #                               truncation=TruncationStrategy.LONGEST_FIRST).data
        #     record["input_ids"] = src['input_ids'] + [self.text_tokenizer.bos_token_id] + tgt['input_ids'][1:]
        #     record["token_type_ids"] = src['token_type_ids'] + [ttid+1 for ttid in tgt['token_type_ids']]
        #     record["attention_mask"] = src['attention_mask'] + tgt['attention_mask']
        # else:
        record = self.text_tokenizer(text=src_text, text_pair=tgt_text, max_length=2*max_length,
                                     padding=PaddingStrategy.MAX_LENGTH, truncation=TruncationStrategy.LONGEST_FIRST).data

        record['labels'] = int(item_label)
        record['src_item_id'] = src_item_id
        record['tgt_item_id'] = tgt_item_id
        record['src_img_emb'] = [float(emb) for emb in src_img_emb.split(",")]
        record['tgt_img_emb'] = [float(emb) for emb in tgt_img_emb.split(",")]

        if self.ensemble == "begin":
            idx_img_1 = record['input_ids'].index(IMG_TOKEN_ID)
            record['image_index'] = record['input_ids'].index(IMG_TOKEN_ID, idx_img_1+1)

        return record

    def __len__(self):
        return len(self.data)


class RobertaImageTwoTowerDataset(Dataset):
    def __init__(self, data, text_tokenizer, max_seq_len, ensemble, max_seq_len_pv=None):
        self.data = data
        self.text_tokenizer = text_tokenizer
        self.max_seq_len = max_seq_len
        self.max_seq_len_pv = max_seq_len_pv
        # self.classification_method = classification_method
        self.ensemble = ensemble

    def __getitem__(self, item):
        record = dict()
        item_label, src_item_id, src_title, src_pvs, src_img_emb, \
            tgt_item_id, tgt_title, tgt_pvs, tgt_img_emb = self.data[item]

        if self.max_seq_len is None:
            src_text = src_pvs
            tgt_text = tgt_pvs
            max_length = self.max_seq_len_pv
        elif self.max_seq_len_pv is None:
            src_text = src_title
            tgt_text = tgt_title
            max_length = self.max_seq_len
        else:
            src_text = " ".join((src_title, self.text_tokenizer.sep_token, " ".join(jieba.cut(src_pvs))))
            tgt_text = " ".join((tgt_title, self.text_tokenizer.sep_token, " ".join(jieba.cut(tgt_pvs))))
            max_length = self.max_seq_len + self.max_seq_len_pv

        if self.ensemble == "begin":
            src_text = " ".join((IMG_TOKEN, self.text_tokenizer.sep_token, src_text))
            tgt_text = " ".join((IMG_TOKEN, self.text_tokenizer.sep_token, tgt_text))

        src = self.text_tokenizer(text=src_text, max_length=max_length, padding=PaddingStrategy.MAX_LENGTH,
                                  truncation=TruncationStrategy.LONGEST_FIRST)
        tgt = self.text_tokenizer(text=tgt_text, max_length=max_length, padding=PaddingStrategy.MAX_LENGTH,
                                  truncation=TruncationStrategy.LONGEST_FIRST)

        # if self.classification_method == "vec_sim":
        #     src = self.text_tokenizer(text=src_text, max_length=max_length, padding=PaddingStrategy.MAX_LENGTH,
        #                               truncation=TruncationStrategy.LONGEST_FIRST).data
        #     tgt = self.text_tokenizer(text=tgt_text, max_length=max_length, padding=PaddingStrategy.MAX_LENGTH,
        #                               truncation=TruncationStrategy.LONGEST_FIRST).data
        #     record["input_ids"] = src['input_ids'] + [self.text_tokenizer.bos_token_id] + tgt['input_ids'][1:]
        #     record["token_type_ids"] = src['token_type_ids'] + [ttid+1 for ttid in tgt['token_type_ids']]
        #     record["attention_mask"] = src['attention_mask'] + tgt['attention_mask']
        # else:
        # record = self.text_tokenizer(text=src_text, text_pair=tgt_text, max_length=2*max_length,
        #                              padding=PaddingStrategy.MAX_LENGTH, truncation=TruncationStrategy.LONGEST_FIRST).data
        src_input_ids = src.data['input_ids']
        src_token_type_ids = src.data['token_type_ids']
        src_attention_mask = src.data['attention_mask']
        tgt_input_ids = tgt.data['input_ids']
        tgt_token_type_ids = tgt.data['token_type_ids']
        tgt_attention_mask = tgt.data['attention_mask']
        record['input_ids_1'] = src_input_ids
        record['input_ids_2'] = tgt_input_ids
        record['token_type_ids_1'] = src_token_type_ids
        record['token_type_ids_2'] = tgt_token_type_ids
        record['attention_mask_1'] = src_attention_mask
        record['attention_mask_2'] = tgt_attention_mask
        # record['cate_ids_1'] = [src_cate_id] * len(src_input_ids)
        # record['cate_ids_2'] = [tgt_cate_id] * len(tgt_input_ids)
        record['labels'] = int(item_label)
        record['src_item_id'] = src_item_id
        record['tgt_item_id'] = tgt_item_id
        record['src_img_emb'] = [float(emb) for emb in src_img_emb.split(",")]
        record['tgt_img_emb'] = [float(emb) for emb in tgt_img_emb.split(",")]
        record['image_index'] = 0

        return record

    def __len__(self):
        return len(self.data)


class RobertaOneTowerPvPairDataset(Dataset):
    def __init__(self, data, text_tokenizer, max_seq_len, classification_method, max_seq_len_pv=None):
        self.data = data
        self.text_tokenizer = text_tokenizer
        self.max_seq_len = max_seq_len
        self.max_seq_len_pv = max_seq_len_pv
        self.classification_method = classification_method

    def __getitem__(self, item):
        item_label, src_item_id, src_title, tgt_item_id, tgt_title, pv_pair_text = self.data[item]
        src_text = src_title
        tgt_text = " ".join((tgt_title, self.text_tokenizer.sep_token, " ".join(jieba.cut(pv_pair_text))))
        max_length = 2*self.max_seq_len + self.max_seq_len_pv

        record = self.text_tokenizer(text=src_text, text_pair=tgt_text, max_length=max_length,
                                     padding=PaddingStrategy.MAX_LENGTH, truncation=TruncationStrategy.LONGEST_FIRST).data

        i1 = record["input_ids"].index(self.text_tokenizer.sep_token_id)
        i2 = record["input_ids"].index(self.text_tokenizer.sep_token_id, i1+1)
        record["token_type_ids"] = record['token_type_ids'][:(i2+1)] + [ttid+1 for ttid in record['token_type_ids'][(i2+1):]]
        record['labels'] = int(item_label)
        record['src_item_id'] = src_item_id
        record['tgt_item_id'] = tgt_item_id

        return record

    def __len__(self):
        return len(self.data)


class RobertaTwoTowerDataset(Dataset):
    def __init__(self, data, text_tokenizer, max_seq_en, max_seq_len_pv=None):
        self.data = data
        self.text_tokenizer = text_tokenizer
        self.max_seq_len_pv = max_seq_len_pv
        self.max_seq_len = max_seq_en

    def __getitem__(self, item):
        record = dict()
        item_label, src_item_id, src_cate_id, src_title, src_pvs, \
            tgt_item_id, tgt_cate_id, tgt_title, tgt_pvs = self.data[item]

        if self.max_seq_len_pv is not None:
            src_text = " ".join((src_title, self.text_tokenizer.sep_token, " ".join(jieba.cut(src_pvs))))
            tgt_text = " ".join((tgt_title, self.text_tokenizer.sep_token, " ".join(jieba.cut(tgt_pvs))))
            max_length = self.max_seq_len + self.max_seq_len_pv
        else:
            src_text = src_title
            tgt_text = tgt_title
            max_length = self.max_seq_len

        src = self.text_tokenizer(text=src_text, max_length=max_length, padding=PaddingStrategy.MAX_LENGTH,
                                  truncation=TruncationStrategy.LONGEST_FIRST)
        tgt = self.text_tokenizer(text=tgt_text, max_length=max_length, padding=PaddingStrategy.MAX_LENGTH,
                                  truncation=TruncationStrategy.LONGEST_FIRST)
        src_input_ids = src.data['input_ids']
        src_token_type_ids = src.data['token_type_ids']
        src_attention_mask = src.data['attention_mask']
        tgt_input_ids = tgt.data['input_ids']
        tgt_token_type_ids = tgt.data['token_type_ids']
        tgt_attention_mask = tgt.data['attention_mask']
        record['input_ids_1'] = src_input_ids
        record['input_ids_2'] = tgt_input_ids
        record['token_type_ids_1'] = src_token_type_ids
        record['token_type_ids_2'] = tgt_token_type_ids
        record['attention_mask_1'] = src_attention_mask
        record['attention_mask_2'] = tgt_attention_mask
        # record['cate_ids_1'] = [src_cate_id] * len(src_input_ids)
        # record['cate_ids_2'] = [tgt_cate_id] * len(tgt_input_ids)
        record['labels'] = int(item_label)
        record['src_item_id'] = src_item_id
        record['tgt_item_id'] = tgt_item_id

        return record

    def __len__(self):
        return len(self.data)


class PairedImageDataset(Dataset):
    def __init__(self, data, input_size, is_training, hflip=0.5, color_jitter=None):
        self.data = data
        self.transform = create_transform(input_size=input_size,
                                          is_training=is_training,
                                          hflip=hflip,
                                          color_jitter=color_jitter)

    def __getitem__(self, item):
        record = dict()
        item_label, src_item_id, src_image_path, tgt_item_id, tgt_image_path = self.data[item]
        # item_label, src_item_id, src_input, tgt_item_id, tgt_input = self.data[item]

        try:
            src_img = Image.open(src_image_path)
            src_img = src_img.convert("RGB")
            src_input = self.transform(src_img)

            tgt_img = Image.open(tgt_image_path)
            tgt_img = tgt_img.convert("RGB")
            tgt_input = self.transform(tgt_img)

            record['src_input'] = src_input
            record['tgt_input'] = tgt_input
        except Exception:
            pass

        record['labels'] = int(item_label)
        record['src_item_id'] = src_item_id
        record['tgt_item_id'] = tgt_item_id

        return record

    def __len__(self):
        return len(self.data)


class MultimodalDataset(Dataset):
    def __init__(self, data, image_size, is_training, text_tokenizer, max_seq_len,
                 max_seq_len_pv=None, hflip=0.5, color_jitter=None):
        self.data = data
        self.transform = create_transform(input_size=image_size,
                                          is_training=is_training,
                                          hflip=hflip,
                                          color_jitter=color_jitter)
        self.text_tokenizer = text_tokenizer
        self.max_seq_len = max_seq_len
        self.max_seq_len_pv = max_seq_len_pv

    def __getitem__(self, item):
        item_id, title, pvs, image_path = self.data[item]

        # text data processing
        if self.max_seq_len is None:
            text = pvs
            max_length = self.max_seq_len_pv
        elif self.max_seq_len_pv is None:
            text = title
            max_length = self.max_seq_len
        else:
            text = " ".join((title, self.text_tokenizer.sep_token, pvs))
            max_length = self.max_seq_len + self.max_seq_len_pv
        text = " ".join((self.text_tokenizer.bos_token, text))
        record = self.text_tokenizer(text=text, max_length=max_length, padding=PaddingStrategy.MAX_LENGTH,
                                     truncation=TruncationStrategy.LONGEST_FIRST).data
        record['position_ids'] = list(range(len(record['input_ids'])))
        # image data processing
        try:
            img = Image.open(image_path)
            img = img.convert("RGB")
            img = self.transform(img)
            record['image'] = img
        except Exception:
            pass

        record['item_id'] = item_id

        return record

    def __len__(self):
        return len(self.data)


class PairedMultimodalDataset(Dataset):
    def __init__(self, data, ensemble, image_size, is_training, text_tokenizer, max_seq_len,
                 max_seq_len_pv=None, hflip=0.5, color_jitter=None):
        self.data = data
        self.transform = create_transform(input_size=image_size,
                                          is_training=is_training,
                                          hflip=hflip,
                                          color_jitter=color_jitter)
        self.ensemble = ensemble
        self.text_tokenizer = text_tokenizer
        self.max_seq_len = max_seq_len
        self.max_seq_len_pv = max_seq_len_pv

    def __getitem__(self, item):
        record = dict()
        item_label, src_item_id, src_title, src_pvs, src_image_path, tgt_item_id, tgt_title, tgt_pvs, tgt_image_path = self.data[item]

        # text data processing
        if self.max_seq_len_pv is not None:
            src_text = " ".join((src_title, self.text_tokenizer.sep_token, " ".join(jieba.cut(src_pvs))))
            tgt_text = " ".join((tgt_title, self.text_tokenizer.sep_token, " ".join(jieba.cut(tgt_pvs))))
            max_length = self.max_seq_len + self.max_seq_len_pv
        else:
            src_text = src_title
            tgt_text = tgt_title
            max_length = self.max_seq_len
        if self.ensemble == "sum":
            src_text = " ".join((self.text_tokenizer.bos_token, src_text))
            tgt_text = " ".join((self.text_tokenizer.bos_token, tgt_text))

        src = self.text_tokenizer(text=src_text, max_length=max_length, padding=PaddingStrategy.MAX_LENGTH,
                                  truncation=TruncationStrategy.LONGEST_FIRST).data
        tgt = self.text_tokenizer(text=tgt_text, max_length=max_length, padding=PaddingStrategy.MAX_LENGTH,
                                  truncation=TruncationStrategy.LONGEST_FIRST).data
        src_input_ids = src['input_ids']
        src_token_type_ids = src['token_type_ids']
        src_attention_mask = src['attention_mask']
        tgt_input_ids = tgt['input_ids']
        tgt_token_type_ids = tgt['token_type_ids']
        tgt_attention_mask = tgt['attention_mask']

        # image data processing
        try:
            src_img = Image.open(src_image_path)
            src_img = src_img.convert("RGB")
            src_img = self.transform(src_img)
            record['src_image'] = src_img
            tgt_img = Image.open(tgt_image_path)
            tgt_img = tgt_img.convert("RGB")
            tgt_img = self.transform(tgt_img)
            record['tgt_image'] = tgt_img
            record['src_image'] = src_img
            record['tgt_image'] = tgt_img
        except Exception:
            pass

        record['src_item_id'] = src_item_id
        record['src_input_ids'] = src_input_ids
        record['src_token_type_ids'] = src_token_type_ids
        record['src_attention_mask'] = src_attention_mask
        record['src_position_ids'] = list(range(len(src_input_ids)))
        record['tgt_input_ids'] = tgt_input_ids
        record['tgt_token_type_ids'] = tgt_token_type_ids
        record['tgt_attention_mask'] = tgt_attention_mask
        record['tgt_item_id'] = tgt_item_id
        record['tgt_position_ids'] = list(range(len(tgt_input_ids)))
        record['labels'] = int(item_label)

        return record

    def __len__(self):
        return len(self.data)


class RobertaDataset(Dataset):
    def __init__(self, data, text_tokenizer, max_seq_len):
        self.data = data
        self.text_tokenizer = text_tokenizer
        self.max_seq_len = max_seq_len

    def __getitem__(self, item):
        # record = dict()
        idx, text = self.data[item]

        # if self.max_seq_len is None:
        #     src_text = src_pvs
        #     tgt_text = tgt_pvs
        #     max_length = self.max_seq_len_pv
        # elif self.max_seq_len_pv is None:
        #     src_text = src_title
        #     tgt_text = tgt_title
        #     max_length = self.max_seq_len
        # else:
        #     src_text = " ".join((src_title, self.text_tokenizer.sep_token, " ".join(jieba.cut(src_pvs))))
        #     tgt_text = " ".join((tgt_title, self.text_tokenizer.sep_token, " ".join(jieba.cut(tgt_pvs))))
        #     max_length = self.max_seq_len + self.max_seq_len_pv

        # if self.classification_method == "vec_sim":
        #     src = self.text_tokenizer(text=src_text, max_length=max_length, padding=PaddingStrategy.MAX_LENGTH,
        #                               truncation=TruncationStrategy.LONGEST_FIRST).data
        #     tgt = self.text_tokenizer(text=tgt_text, max_length=max_length, padding=PaddingStrategy.MAX_LENGTH,
        #                               truncation=TruncationStrategy.LONGEST_FIRST).data
        #     record["input_ids"] = src['input_ids'] + [self.text_tokenizer.bos_token_id] + tgt['input_ids'][1:]
        #     record["token_type_ids"] = src['token_type_ids'] + [ttid+1 for ttid in tgt['token_type_ids']]
        #     record["attention_mask"] = src['attention_mask'] + tgt['attention_mask']
        #     len_src = len(src['input_ids'])
        #     len_tgt = len(tgt['input_ids'])
        # else:
        record = self.text_tokenizer(text=text, max_length=self.max_seq_len,
                                     padding=PaddingStrategy.MAX_LENGTH, truncation=TruncationStrategy.LONGEST_FIRST).data
        record['idx'] = idx

        return record

    def __len__(self):
        return len(self.data)


class GCNDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)