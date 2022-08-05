
import torch

from typing import List, Optional, Tuple, Union
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from transformers import RobertaPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaPooler, \
    BaseModelOutputWithPoolingAndCrossAttentions
from .loss import EuclideanDistanceLoss, HingeLoss
from .base import (
    RobertaEmbeddings,
    RobertaImageEmbeddings,
    VecSimClassificationHead,
    RobertaClassificationHead,
    SequenceClassifierOutput,
    TwoTowerClassificationHead
)


# Roberta + Image model
class RobertaImageModel(RobertaPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        if config.ensemble == "begin":
            self.embeddings = RobertaImageEmbeddings(config)
        else:
            self.embeddings = RobertaEmbeddings(config)

        self.encoder = RobertaEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[Union[torch.Tensor, List]] = None,
            image_indices: Optional[List] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None:
            input_shape = input_ids.size()
        # elif inputs_embeds is not None:
        #     input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if self.config.ensemble == "begin":
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
                attention_mask=attention_mask,
                image_indices=image_indices
            )
        else:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=None,
                past_key_values_length=past_key_values_length
            )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class RobertaImageOneTower(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        if config.max_seq_len_pv is None:
            self.max_seq_len = config.max_seq_len
        elif config.max_seq_len is None:
            self.max_seq_len = config.max_seq_len_pv
        else:
            self.max_seq_len = config.max_seq_len + config.max_seq_len_pv
        self.cls_layers = [-int(i) for i in config.cls_layers.split(",")]

        self.roberta = RobertaImageModel(config, add_pooling_layer=False)
        if config.classification_method == "vec_sim":
            self.classifier = VecSimClassificationHead(config)
        else:
            self.classifier = RobertaClassificationHead(config)

        if config.loss_type == "cosine":
            self.loss_fct = nn.CosineEmbeddingLoss(margin=config.loss_margin)
        elif config.loss_type == "bce":
            self.loss_fct = nn.BCEWithLogitsLoss()
        elif config.loss_type == "euclidean":
            self.loss_fct = EuclideanDistanceLoss()
        elif config.loss_type == "hinge":
            self.loss_fct = HingeLoss(margin=config.loss_margin)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        self.softmax = torch.nn.Softmax()
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[Union[torch.Tensor, List]] = None,
            image_indices: Optional[List] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_indices=image_indices,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_outputs = [outputs.hidden_states[i] for i in self.cls_layers]
        if self.config.cls_pool == "avg":
            sequence_output = torch.stack(sequence_outputs).mean(axis=0)
        else:
            sequence_output = torch.cat(sequence_outputs, dim=-1)
        if self.config.classification_method == "vec_sim":
            src_sequence_output = sequence_output[:, 0, :]
            tgt_sequence_output = sequence_output[:, self.max_seq_len, :]
            src_embeds, tgt_embeds, logits, probs = self.classifier(src_sequence_output, tgt_sequence_output)
        else:
            logits = self.classifier(sequence_output, inputs_embeds=inputs_embeds)
            probs = self.softmax(logits)
            src_embeds = probs[:, 0]
            tgt_embeds = probs[:, 1]
            probs = probs[:, 1]

        loss = None
        if labels is not None:
            if self.config.loss_type == "cosine":
                loss = self.loss_fct(src_embeds, tgt_embeds, (labels*2-1).view(-1))
            elif self.config.loss_type == "ce":
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.loss_type == "hinge" or self.config.loss_type == "euclidean":
                loss = self.loss_fct(logits.view(-1), (labels*2-1).view(-1))
            else:
                loss = self.loss_fct(logits.view(-1), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            probs=probs,
            src_embeds=src_embeds,
            tgt_embeds=tgt_embeds
        )


class RobertaImageTwoTower(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        if config.max_seq_len_pv is None:
            self.max_seq_len = config.max_seq_len
        elif config.max_seq_len is None:
            self.max_seq_len = config.max_seq_len_pv
        else:
            self.max_seq_len = config.max_seq_len + config.max_seq_len_pv
        self.cls_layers = [-int(i) for i in config.cls_layers.split(",")]

        self.roberta = RobertaImageModel(config, add_pooling_layer=False)
        # if config.classification_method == "vec_sim":
        #     self.classifier = VecSimClassificationHead(config)
        # else:
        #     self.classifier = RobertaClassificationHead(config)
        self.classifier = TwoTowerClassificationHead(config.hidden_size,
                                                     dropout=config.hidden_dropout_prob,
                                                     num_labels=config.num_labels)

        if config.loss_type == "cosine":
            self.loss_fct = nn.CosineEmbeddingLoss(margin=config.loss_margin)
        elif config.loss_type == "bce":
            self.loss_fct = nn.BCEWithLogitsLoss()
        elif config.loss_type == "euclidean":
            self.loss_fct = EuclideanDistanceLoss()
        elif config.loss_type == "hinge":
            self.loss_fct = HingeLoss(margin=config.loss_margin)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        self.softmax = torch.nn.Softmax()
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids_1: Optional[torch.LongTensor] = None,
            attention_mask_1: Optional[torch.FloatTensor] = None,
            token_type_ids_1: Optional[torch.LongTensor] = None,
            position_ids_1: Optional[torch.LongTensor] = None,
            images_1: Optional[torch.FloatTensor] = None,
            input_ids_2: Optional[torch.LongTensor] = None,
            attention_mask_2: Optional[torch.FloatTensor] = None,
            token_type_ids_2: Optional[torch.LongTensor] = None,
            position_ids_2: Optional[torch.LongTensor] = None,
            images_2: Optional[torch.FloatTensor] = None,
            # image_indices: Optional[List] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        image_indices = torch.zeros(input_ids_1.shape, dtype=torch.long, device=input_ids_1.device)
        outputs_1 = self.roberta(
            input_ids_1,
            attention_mask=attention_mask_1,
            token_type_ids=token_type_ids_1,
            position_ids=position_ids_1,
            head_mask=head_mask,
            inputs_embeds=images_1,
            image_indices=image_indices,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output_1 = outputs_1.last_hidden_state
        # sequence_outputs_1 = [outputs_1.hidden_states[i] for i in self.cls_layers]
        # if self.config.cls_pool == "avg":
        #     sequence_output_1 = torch.stack(sequence_outputs_1).mean(axis=0)
        # else:
        #     sequence_output_1 = torch.cat(sequence_outputs_1, dim=-1)

        outputs_2 = self.roberta(
            input_ids_2,
            attention_mask=attention_mask_2,
            token_type_ids=token_type_ids_2,
            position_ids=position_ids_2,
            head_mask=head_mask,
            inputs_embeds=images_2,
            image_indices=image_indices,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output_2 = outputs_2.last_hidden_state
        # sequence_outputs_2 = [outputs_2.hidden_states[i] for i in self.cls_layers]
        # if self.config.cls_pool == "avg":
        #     sequence_output_2 = torch.stack(sequence_outputs_2).mean(axis=0)
        # else:
        #     sequence_output_2 = torch.cat(sequence_outputs_2, dim=-1)

        # if self.config.classification_method == "vec_sim":
        #     src_sequence_output = sequence_output[:, 0, :]
        #     tgt_sequence_output = sequence_output[:, self.max_seq_len, :]
        #     src_embeds, tgt_embeds, logits, probs = self.classifier(src_sequence_output, tgt_sequence_output)
        # else:
        #     logits = self.classifier(sequence_output, inputs_embeds=inputs_embeds)
        #     probs = self.softmax(logits)
        #     src_embeds = probs[:, 0]
        #     tgt_embeds = probs[:, 1]
        #     probs = probs[:, 1]

        src_embeds, tgt_embeds, logits, probs = self.classifier(sequence_output_1[:, 0, :], sequence_output_2[:, 0, :])

        loss = None
        if labels is not None:
            if self.config.loss_type == "cosine":
                loss = self.loss_fct(src_embeds, tgt_embeds, (labels*2-1).view(-1))
            elif self.config.loss_type == "ce":
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.loss_type == "hinge" or self.config.loss_type == "euclidean":
                loss = self.loss_fct(logits.view(-1), (labels*2-1).view(-1))
            else:
                loss = self.loss_fct(logits.view(-1), labels.view(-1))

        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            probs=probs,
            src_embeds=src_embeds,
            tgt_embeds=tgt_embeds
        )


# CoCa
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# normalization
# they use layernorm without bias, something that pytorch does not offer
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


# residual
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# rotary positional embedding
# https://arxiv.org/abs/2104.09864
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame
class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4, is_decoding=False):
        super().__init__()
        self.is_decoding = is_decoding

        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # for caching causal mask and rotary embeddings
        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    def get_upper_triangular_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x, attn_mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm
        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner
        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # rotary embeddings
        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale
        q = q * self.scale

        # similarity
        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # causal mask
        if self.is_decoding:
            causal_mask = self.get_upper_triangular_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # extra attention mask - for masking out attention from text CLS token to padding
        if exists(attn_mask):
            attn_mask = rearrange(attn_mask, 'b i j -> b 1 i j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        # attention
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values
        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.attn_out(out) + self.ff_out(ff)


# cross attention - using multi-query + one-headed key / values as in PaLM w/ optional parallel feedforward
class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            *,
            context_dim=None,
            dim_head=64,
            heads=8,
            parallel_ff=False,
            ff_mult=4,
            norm_context=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        ) if parallel_ff else None

    def forward(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # pre-layernorm, for queries and context
        x = self.norm(x)
        context = self.context_norm(context)

        # get queries
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        # scale
        q = q * self.scale

        # get key / values
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # query / key similarity
        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # attention
        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        # aggregate
        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # add parallel feedforward (for multimodal layers)
        if exists(self.ff):
            out = out + self.ff(x)

        return out


class CoCaModel(nn.Module):
    def __init__(
            self,
            config,
            # image_dim = None,
            # num_img_queries=256,
            # dim_head=64,
            # heads=8,
            # ff_mult=4,
            image_encoder=None,
            text_encoder=None,
            # additional_image_query=True
    ):
        super().__init__()
        self.config = config
        # self.dim = config.hidden_size
        # self.additional_image_query = additional_image_query

        # token embeddings
        # self.token_emb = nn.Embedding(num_tokens, dim)
        # self.text_cls_token = nn.Parameter(torch.randn(dim))

        # image encoder
        self.img_encoder = image_encoder

        # # attention pooling for image tokens
        # num_image_queries = config.num_image_queries + (1 if additional_image_query else 0)
        # self.img_queries = nn.Parameter(torch.randn(num_image_queries, config.hidden_size)) # num image queries for multimodal, but 1 extra CLS for contrastive learning
        # hidden_size_head = config.hidden_size // config.num_attention_heads
        # self.img_attn_pool = CrossAttention(dim=config.hidden_size, context_dim=config.hidden_size,
        #                                     dim_head=hidden_size_head, heads=config.num_attention_heads,
        #                                     norm_context=True)
        # self.img_attn_pool_norm = LayerNorm(config.hidden_size)

        # text encoder
        self.text_encoder = text_encoder
        # self.text_cls_norm = LayerNorm(config.hidden_size)
        # self.unimodal_layers = nn.ModuleList([])
        # for ind in range(unimodal_depth):
        #     self.unimodal_layers.append(
        #         Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
        #     )

    # def embed_text(self, text):
    #     batch, device = text.shape[0], text.device
    #
    #     seq = text.shape[1]
    #
    #     text_tokens = self.token_emb(text)
    #
    #     # append text cls tokens
    #
    #     text_cls_tokens = repeat(self.text_cls_token, 'd -> b 1 d', b=batch)
    #     text_tokens = torch.cat((text_tokens, text_cls_tokens), dim=-2)
    #
    #     # create specific mask for text cls token at the end
    #     # to prevent it from attending to padding
    #
    #     cls_mask = rearrange(text != self.pad_id, 'b j -> b 1 j')
    #     attn_mask = F.pad(cls_mask, (0, 1, seq, 0), value=True)
    #
    #     # go through unimodal layers
    #
    #     for attn_ff in self.unimodal_layers:
    #         text_tokens = attn_ff(text_tokens, attn_mask=attn_mask)
    #
    #     # get text cls token
    #
    #     text_tokens, text_cls_tokens = text_tokens[:, :-1], text_tokens[:, -1]
    #     text_embeds = self.text_cls_norm(text_cls_tokens)
    #     return text_embeds, text_tokens

    def embed_text(self, input_ids, attention_mask, token_type_ids, position_ids):
        output = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        sequence_output = output.last_hidden_state

        text_embeds = sequence_output[:, 0]
        # if self.additional_image_query:
        #     sequence_output = sequence_output[:, 1:]

        # text_embeds = self.text_cls_norm(text_cls_tokens)

        return text_embeds, sequence_output

    def embed_image(self, images=None, image_tokens=None):
        # encode images into embeddings
        # with the img_encoder passed in at init
        # it can also accept precomputed image tokens

        assert not (exists(images) and exists(image_tokens))

        image_embeds = None
        image_sequence_output = None
        if exists(images):
            assert exists(self.img_encoder), 'img_encoder must be passed in for automatic image encoding'
            # image_embeds, image_sequence_output = self.img_encoder(images)
            image_sequence_output = self.img_encoder.forward_features(images)
            image_embeds = self.img_encoder.forward_head(image_sequence_output, pre_logits=True)

        # # attention pool image tokens
        # img_queries = repeat(self.img_queries, 'n d -> b n d', b=image_tokens.shape[0])
        # img_queries = self.img_attn_pool(img_queries, image_tokens)
        # image_sequence_output = self.img_attn_pool_norm(img_queries)
        #
        # if self.additional_image_query:
        #     image_embeds = image_sequence_output[:, 0]
        #     image_sequence_output = image_sequence_output[:, 1:]
        # else:
        #     image_embeds = None

        return image_embeds, image_sequence_output

    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            images,
            image_tokens=None,
    ):
        text_embeds, text_tokens = self.embed_text(input_ids, attention_mask, token_type_ids, position_ids)

        image_embeds, image_tokens = self.embed_image(images=images, image_tokens=image_tokens)

        return text_embeds, text_tokens, image_embeds, image_tokens


class CoCaForPretraining(nn.Module):
    def __init__(
            self,
            config,
            image_encoder=None,
            text_encoder=None,
            pad_id=0
    ):
        super().__init__()
        self.config = config
        # self.dim = config.hidden_size
        self.pad_id = pad_id
        self.caption_loss_weight = config.caption_loss_weight
        self.contrastive_loss_weight = config.contrastive_loss_weight

        self.coca = CoCaModel(config, image_encoder, text_encoder)

        # contrastive learning temperature
        self.temperature = nn.Parameter(torch.Tensor([1.]))

        # multimodal layers
        self.multimodal_layers = nn.ModuleList([])
        hidden_size_head_multimodal = config.hidden_size // config.num_attention_heads_multimodal
        for ind in range(config.num_hidden_layers_multimodal):
            self.multimodal_layers.append(nn.ModuleList([
                Residual(ParallelTransformerBlock(dim=config.hidden_size, dim_head=hidden_size_head_multimodal,
                                                  heads=config.num_attention_heads_multimodal,
                                                  ff_mult=config.feedforward_multiplication_multimodal, is_decoding=True)),
                Residual(CrossAttention(dim=config.hidden_size, dim_head=hidden_size_head_multimodal,
                                        heads=config.num_attention_heads_multimodal, parallel_ff=True,
                                        ff_mult=config.feedforward_multiplication_multimodal))
            ]))

        # to logits
        self.to_logits = nn.Sequential(
            LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        )

        # they used embedding weight tied projection out to logits, not common, but works
        self.to_logits[-1].weight = self.coca.text_encoder.embeddings.word_embeddings.weight
        # self.to_logits[-1].weight = self.token_emb.weight
        # nn.init.normal_(self.token_emb.weight, std=0.02)

    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            images=None,
            image_tokens=None,
            labels=None
    ):
        batch, device = input_ids.shape[0], input_ids.device

        if not exists(labels):
            labels = input_ids[:, 2:]
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]
            token_type_ids = token_type_ids[:, :-1]
            position_ids = position_ids[:, :-1]

        text_embeds, text_tokens, image_embeds, image_tokens = self.coca(input_ids, attention_mask,
                                                                         token_type_ids, position_ids,
                                                                         images, image_tokens)

        # go through multimodal layers
        for attn_ff, cross_attn in self.multimodal_layers:
            text_tokens = attn_ff(text_tokens)
            text_tokens = cross_attn(text_tokens, image_tokens)

        logits = self.to_logits(text_tokens)

        # shorthand
        ce = F.cross_entropy

        # calculate caption loss (cross entropy loss)
        logits = rearrange(logits, 'b n c -> b c n')
        caption_loss = ce(logits, labels, ignore_index=self.pad_id)
        caption_loss = caption_loss * self.caption_loss_weight

        # calculate contrastive loss
        sim = einsum('i d, j d -> i j', text_embeds, image_embeds)
        sim = sim * self.temperature.exp()
        contrastive_labels = torch.arange(batch, device=device)

        contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5
        contrastive_loss = contrastive_loss * self.contrastive_loss_weight

        return caption_loss + contrastive_loss


class CoCaForItemAlignment(nn.Module):
    def __init__(
            self,
            config,
            image_encoder=None,
            text_encoder=None
    ):
        super().__init__()
        self.config = config
        # self.dim = config.hidden_size
        self.ensemble = config.ensemble
        self.num_labels = config.num_labels
        # additional_image_query = True if config.ensemble == "sum" else False
        self.coca = CoCaModel(config, image_encoder, text_encoder)

        if config.ensemble == "cross_attn":
            # multimodal layers
            self.multimodal_layers = nn.ModuleList([])
            hidden_size_head_multimodal = config.hidden_size // config.num_attention_heads_multimodal
            for ind in range(config.num_hidden_layers_multimodal):
                self.multimodal_layers.append(nn.ModuleList([
                    Residual(ParallelTransformerBlock(dim=config.hidden_size, dim_head=hidden_size_head_multimodal,
                                                      heads=config.num_attention_heads_multimodal,
                                                      ff_mult=config.feedforward_multiplication_multimodal, is_decoding=False)),
                    Residual(CrossAttention(dim=config.hidden_size, dim_head=hidden_size_head_multimodal,
                                            heads=config.num_attention_heads_multimodal, parallel_ff=True,
                                            ff_mult=config.feedforward_multiplication_multimodal))
                ]))

        if config.classification_method == "vec_sim":
            self.classifier = VecSimClassificationHead(config)
        else:
            self.classifier = TwoTowerClassificationHead(config.hidden_size,
                                                         dropout=config.hidden_dropout_prob,
                                                         num_labels=config.num_labels)

        if config.loss_type == "cosine":
            self.loss_fct = nn.CosineEmbeddingLoss(margin=config.loss_margin)
        elif config.loss_type == "bce":
            self.loss_fct = nn.BCEWithLogitsLoss()
        elif config.loss_type == "euclidean":
            self.loss_fct = EuclideanDistanceLoss()
        elif config.loss_type == "hinge":
            self.loss_fct = HingeLoss(margin=config.loss_margin)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids_1,
            attention_mask_1,
            token_type_ids_1,
            position_ids_1,
            images_1,
            input_ids_2,
            attention_mask_2,
            token_type_ids_2,
            position_ids_2,
            images_2,
            labels=None
    ):
        text_embeds_1, text_tokens_1, image_embeds_1, image_tokens_1 = \
            self.coca(input_ids_1, attention_mask_1, token_type_ids_1, position_ids_1, images_1)

        text_embeds_2, text_tokens_2, image_embeds_2, image_tokens_2 = \
            self.coca(input_ids_2, attention_mask_2, token_type_ids_2, position_ids_2, images_2)

        if self.ensemble == "cross_attn":
            # go through multimodal layers
            for attn_ff, cross_attn in self.multimodal_layers:
                text_tokens_1 = attn_ff(text_tokens_1)
                text_tokens_1 = cross_attn(text_tokens_1, image_tokens_1)
            embeds_1 = text_tokens_1[:, 0]

            for attn_ff, cross_attn in self.multimodal_layers:
                text_tokens_2 = attn_ff(text_tokens_2)
                text_tokens_2 = cross_attn(text_tokens_2, image_tokens_2)
            embeds_2 = text_tokens_1[:, 0]
        else:
            embeds_1 = text_embeds_1 + image_embeds_1
            embeds_2 = text_embeds_2 + image_embeds_2

        # calculate loss (cross entropy loss)
        src_embeds, tgt_embeds, logits, probs = self.classifier(embeds_1, embeds_2)
        src_embeds = probs[:, 0]
        tgt_embeds = probs[:, 1]
        probs = probs[:, 1]

        loss = None
        if labels is not None:
            if self.config.loss_type == "cosine":
                loss = self.loss_fct(src_embeds, tgt_embeds, (labels*2-1).view(-1))
            elif self.config.loss_type == "ce":
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.loss_type == "hinge" or self.config.loss_type == "euclidean":
                loss = self.loss_fct(logits.view(-1), (labels*2-1).view(-1))
            else:
                loss = self.loss_fct(logits.view(-1), labels.view(-1))

        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            probs=probs,
            logits=logits,
            src_embeds=src_embeds,
            tgt_embeds=tgt_embeds
        )
