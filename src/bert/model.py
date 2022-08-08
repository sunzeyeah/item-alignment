# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :EntityAlignNet
# @File     :bert
# @Date     :2022/6/29 11:10
# @Author   :mengqingyang
# @Email    :mengqingyang0102@163.com
-------------------------------------------------
"""
import torch
from torch import nn
from pytorch_transformers import BertPreTrainedModel
from pytorch_transformers.modeling_bert import BertEncoder, BertPooler, BertOnlyNSPHead, BertLayerNorm
from torch.nn import CrossEntropyLoss


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)


        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModel(BertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids,
                token_type_ids=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                noise=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids,
                                           position_ids=position_ids,
                                           token_type_ids=token_type_ids)
        if noise is not None:
            embedding_output = embedding_output + noise
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertAlignModel(BertPreTrainedModel):
    def __init__(self, config=None):
        """title, cate, cate_path, pvs, industry_name"""
        super(BertAlignModel, self).__init__(config)
        self.loss_fct = CrossEntropyLoss()
        self.num_labels = 2
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.init_weights()

    def get_sim_eval_weight(self):
        w = self.cls.seq_relationship.weight.detach().cpu()
        b = self.cls.seq_relationship.bias.detach().cpu()
        weight = w[1] - w[0]
        return weight, b[1] - b[0]

    def forward(self,
                pvs_input_ids=None, pvs_token_type_ids=None, pvs_attention_mask=None,
                title_input_ids=None, title_token_type_ids=None, title_attention_mask=None,
                cate_input_ids=None, cate_token_type_ids=None, cate_attention_mask=None,
                cate_path_input_ids=None, cate_path_token_type_ids=None, cate_path_attention_mask=None,
                industry_name_input_ids=None, industry_name_token_type_ids=None, industry_name_attention_mask=None,
                next_sentence_label=None, pvs_noise=None, title_noise=None):
        title_out = self.bert(title_input_ids,
                              token_type_ids=title_token_type_ids,
                              attention_mask=title_attention_mask,
                              noise=title_noise)
        title_pool_out = title_out[1]

        cate_out = self.bert(cate_input_ids,
                             token_type_ids=cate_token_type_ids,
                             attention_mask=cate_attention_mask)
        cate_pool_out = cate_out[1]

        cate_path_out = self.bert(cate_path_input_ids,
                                  token_type_ids=cate_path_token_type_ids,
                                  attention_mask=cate_path_attention_mask)
        cate_path_pool_out = cate_path_out[1]

        pvs_out = self.bert(pvs_input_ids,
                            token_type_ids=pvs_token_type_ids,
                            attention_mask=pvs_attention_mask,
                            noise=pvs_noise)
        pvs_pool_out = pvs_out[1]

        industry_name_out = self.bert(industry_name_input_ids,
                                      token_type_ids=industry_name_token_type_ids,
                                      attention_mask=industry_name_attention_mask)
        industry_name_pool_out = industry_name_out[1]

        pool_out = title_pool_out + cate_pool_out + cate_path_pool_out + pvs_pool_out + industry_name_pool_out
        seq_relationship_score = self.cls(pool_out)
        output = [pool_out, seq_relationship_score]
        if next_sentence_label is not None:
            next_sentence_loss = self.loss_fct(seq_relationship_score.view(-1, self.num_labels),
                                               next_sentence_label.view(-1))
            output.append(next_sentence_loss)
        return output

