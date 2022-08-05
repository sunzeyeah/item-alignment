
import torch

from packaging import version
from torch import nn
from typing import List, Optional, Tuple, Union
from transformers.utils import ModelOutput


class InnerProduct(nn.Module):
    r"""
    Computes the innner product between vectors :math:`v_1`, :math:`v_2`

    Args:
        normalize (bool, optional): whether to normalize vector before computi. Default: False
    Shape:
        - Input1: :math:`(N, D)` or :math:`(D)` where `N = batch dimension` and `D = vector dimension`
        - Input2: :math:`(N, D)` or :math:`(D)`, same shape as the Input1
        - Output: :math:`(N)` or :math:`()` based on input dimension.
          If :attr:`keepdim` is ``True``, then :math:`(N, 1)` or :math:`(1)` based on input dimension.
    """
    __constants__ = ['normalize']
    normalize: bool

    def __init__(self, normalize: bool = False) -> None:
        super(InnerProduct, self).__init__()
        self.normalize = normalize

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        bs, hs = x1.shape
        if self.normalize:
            x1 = nn.functional.normalize(x1, p=2, dim=1)
            x2 = nn.functional.normalize(x2, p=2, dim=1)
        return torch.bmm(x1.view(bs, 1, hs), x2.view(bs, hs, 1)).reshape(-1)


class VecSimClassificationHead(nn.Module):
    ''' Because input sequence consists of 4 parts:
     [CLS] src_text_ids [SEP] src_kg_ids [CLS] tgt_text_ids [SEP] tgt_kg_ids

    Therefore choose the 2 [CLS] tokens as representation of two sequences respectively
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config
        cls_layers = [int(i) for i in config.cls_layers.split(",")]
        length = 1 if config.cls_pool == "avg" else len(cls_layers)
        self.dense = nn.Linear(config.hidden_size * length, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        if config.similarity_measure == "inner_product":
            self.similarity = InnerProduct(normalize=False)
            self.sigmoid = nn.Sigmoid()
        elif config.similarity_measure == "cosine":
            self.similarity = nn.CosineSimilarity()
        elif config.similarity_measure == "l1":
            self.similarity = nn.PairwiseDistance(p=1)
        elif config.similarity_measure == "l2":
            self.similarity = nn.PairwiseDistance(p=2)
        else:
            raise ValueError(f"Unsupported similarty measure: {config.similarity_measure}")

    def forward(self, features_1, features_2):
        x = self.dropout(features_1)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        y = self.dropout(features_2)
        y = self.dense(y)
        y = torch.tanh(y)
        y = self.dropout(y)

        sim = self.similarity(x, y)

        if self.config.similarity_measure == "cosine":
            probs = (sim + 1) / 2
        elif self.config.similarity_measure == "l1" or self.config.similarity_measure == "l2":
            probs = torch.exp(-sim)
        elif self.config.similarity_measure == "inner_product":
            probs = self.sigmoid(sim)
        else:
            raise ValueError(f"Unsupported similarty measure: {self.config.similarity_measure}")

        return x, y, sim, probs


class TwoTowerClassificationHead(nn.Module):
    """Head for two-tower sentence-level classification tasks.
    concat the output vector of two tower models and project to num_classes
    """

    def __init__(self, hidden_size, dropout=0.0, num_labels=2):
        super().__init__()
        # self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size * 2, num_labels)
        self.softmax = torch.nn.Softmax()

    def forward(self, features_1, features_2):
        x = self.dropout(features_1)
        # x = self.dense(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)

        y = self.dropout(features_2)
        # y = self.dense(y)
        # y = torch.tanh(y)
        # y = self.dropout(y)

        logits = self.out_proj(torch.cat((x, y), dim=1))
        probs = self.softmax(logits)

        return x, y, logits, probs


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        cls_layers = [int(i) for i in config.cls_layers.split(",")]
        length = 1 if config.cls_pool == "avg" else len(cls_layers)
        self.dense = nn.Linear(config.hidden_size * length, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        if config.ensemble is not None and config.ensemble == "end":
            self.dense_img = nn.Linear(2*config.image_hidden_size, config.hidden_size)
            self.out_proj = nn.Linear(2*config.hidden_size, config.num_labels)
        else:
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        if self.config.ensemble is not None and self.config.ensemble == "end":
            image_embeds = kwargs.get("inputs_embeds")
            y = torch.cat(image_embeds, dim=-1)
            y = self.dropout(y)
            y = self.dense_img(y)
            y = torch.tanh(y)
            y = self.dropout(y)
            x = self.out_proj(torch.cat((x, y), dim=-1))
        else:
            x = self.out_proj(x)

        return x


class SequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    src_embeds: Optional[torch.FloatTensor] = None
    tgt_embeds: Optional[torch.FloatTensor] = None
    probs: torch.FloatTensor = None
    # logits: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


class RobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.cate_embeddings = nn.Embedding(config.cate_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, cate_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        if cate_ids is not None:
            cate_embeddings = self.cate_embeddings(cate_ids)
            embeddings += cate_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class RobertaPKGMEmbeddings(nn.Module):
    """ RobertaPKGMEmbedding: input sequence consists of 4 parts:
     [CLS] src_text_ids [SEP] src_kg_ids [CLS] tgt_text_ids [SEP] tgt_kg_ids

     use RobertaEmbedding to encode text ids, and KGEmbedding to encode kg ids.
     Concatenate roberta embeddings and kg embeddings. Then add position and token type embeddings
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.ent_emb = nn.Embedding(config.num_entities, config.kg_embedding_dim)
        self.rel_emb = nn.Embedding(config.num_relations, config.kg_embedding_dim)
        self.proj_mat = nn.Linear(config.kg_embedding_dim, config.kg_embedding_dim,
                                  bias=config.entity_projection_bias)
        if config.kg_embedding_dim != config.hidden_size:
            self.entity_embedding_projetor = nn.Linear(config.kg_embedding_dim, config.hidden_size)
            self.relation_embedding_projetor = nn.Linear(config.kg_embedding_dim, config.hidden_size)
            self.entity_projection_projetor = nn.Linear(config.kg_embedding_dim, config.hidden_size)
        else:
            self.entity_embedding_projetor = None
            self.relation_embedding_projetor = None
            self.entity_projection_projetor = None

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        # if version.parse(torch.__version__) > version.parse("1.6.0"):
        #     self.register_buffer(
        #         "token_type_ids",
        #         torch.zeros(self.position_ids.size(), dtype=torch.long),
        #         persistent=False,
        #     )

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def kg_embeddings(self, input_ids):
        ''' kg embeddings consists of 2 parts:
        (1) triple query embedding = h+r
        (2) relation query embedding = M*h-r

        :param input_ids:
        :return:
        '''
        src_entity_ids = input_ids[:, self.config.max_seq_len].unsqueeze(1)
        src_relation_ids = input_ids[:, (self.config.max_seq_len+1):(self.config.max_pvs+self.config.max_seq_len+1)]
        src_h = nn.functional.normalize(self.ent_emb(src_entity_ids))
        src_r = self.rel_emb(src_relation_ids)
        src_h_projected = self.proj_mat(src_h)
        if self.config.interaction_type == "one_tower":
            tgt_entity_ids = input_ids[:, 2*self.config.max_seq_len+self.config.max_pvs+1].unsqueeze(1)
            tgt_relation_ids = input_ids[:, (2*self.config.max_seq_len+self.config.max_pvs+2):]
            tgt_h = nn.functional.normalize(self.ent_emb(tgt_entity_ids))
            tgt_r = self.rel_emb(tgt_relation_ids)
            tgt_h_projected = self.proj_mat(tgt_h)
        if self.entity_embedding_projetor is not None:
            src_h = self.entity_embedding_projetor(src_h)
            if self.config.interaction_type == "one_tower":
                tgt_h = self.entity_embedding_projetor(tgt_h)
        if self.relation_embedding_projetor is not None:
            src_r = self.relation_embedding_projetor(src_r)
            if self.config.interaction_type == "one_tower":
                tgt_r = self.relation_embedding_projetor(tgt_r)
        if self.entity_projection_projetor is not None:
            src_h_projected = self.entity_projection_projetor(src_h_projected)
            if self.config.interaction_type == "one_tower":
                tgt_h_projected = self.entity_projection_projetor(tgt_h_projected)

        # triple query module= h+r
        src_triple_query = src_h + src_r
        if self.config.interaction_type == "one_tower":
            tgt_triple_query = tgt_h + tgt_r

        # relation query module= M*h-r
        src_relation_query = src_h_projected - src_r
        if self.config.interaction_type == "one_tower":
            tgt_relation_query = tgt_h_projected - tgt_r

        return (
            torch.cat((src_triple_query, src_relation_query), dim=1),
            torch.cat((tgt_triple_query, tgt_relation_query), dim=1) if self.config.interaction_type == "one_tower" else None
        )

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            src_text_ids = input_ids[:, :self.config.max_seq_len]
            src_text_embeds = self.word_embeddings(src_text_ids)
            if self.config.interaction_type == "one_tower":
                tgt_text_ids = input_ids[:, (self.config.max_seq_len + self.config.max_pvs + 1):(2*self.config.max_seq_len + self.config.max_pvs + 1)]
                tgt_text_embeds = self.word_embeddings(tgt_text_ids)
            src_kg_embeds, tgt_kg_embeds = self.kg_embeddings(input_ids)
            if self.config.interaction_type == "one_tower":
                inputs_embeds = torch.cat((src_text_embeds, src_kg_embeds, tgt_text_embeds, tgt_kg_embeds), dim=1)
            else:
                inputs_embeds = torch.cat((src_text_embeds, src_kg_embeds), dim=1)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class RobertaImageEmbeddings(nn.Module):
    """
    Same as RoBertaEmbeddings with image embedding provided

    [CLS] [IMG] token_1 ... [SEP] [IMG] token_k ... [SEP]

    where [IMG] represent token for image, but image embedding is provided as inputs_embeds (batch_size * 2 * img_emb_size)
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        if config.ensemble == "begin":
            self.img2txt = nn.Linear(config.image_hidden_size, config.hidden_size, bias=True)

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, attention_mask=None,
            image_indices=None, past_key_values_length=0
    ):
        if position_ids is None:
            if attention_mask is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(attention_mask, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        inputs_embeds_txt = self.word_embeddings(input_ids)
        if self.config.ensemble == "begin":
            if self.config.interaction_type == "one_tower":
                inputs_embeds = self.img2txt(torch.stack(inputs_embeds, dim=1))
                inputs_embeds_final = []
                for i, img_idx in enumerate(image_indices):
                    inputs_embeds_final.append(torch.cat((inputs_embeds_txt[i][0:1],
                                                          inputs_embeds[i][0:1],
                                                          inputs_embeds_txt[i][2:img_idx],
                                                          inputs_embeds[i][1:2],
                                                          inputs_embeds_txt[i][img_idx+1:]), dim=0))
                inputs_embeds = torch.stack(inputs_embeds_final)
            else:
                inputs_embeds = self.img2txt(inputs_embeds)
                inputs_embeds = torch.cat([inputs_embeds_txt[:, 0:1, :],
                                           inputs_embeds.unsqueeze(1),
                                           inputs_embeds_txt[:, 2:, :]], dim=1)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


