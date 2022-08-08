
import os
import torch

from collections import OrderedDict
from typing import List, Optional, Tuple, Union
from torch import nn
import torch.nn.functional as F
from transformers import (
    PretrainedConfig,
    BertConfig,
    # RobertaModel,
    RobertaPreTrainedModel
)
from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaPooler, \
    BaseModelOutputWithPoolingAndCrossAttentions
from ..utils import logger, ROBERTA_WEIGHTS_NAME, KG_WEIGHTS_NAME
from .base import (
    RobertaEmbeddings,
    RobertaPKGMEmbeddings,
    TwoTowerClassificationHead,
    VecSimClassificationHead,
    RobertaClassificationHead,
    SequenceClassifierOutput,
)
from .loss import HingeLoss, EuclideanDistanceLoss


# class RobertaPKGMConfig(BertConfig):
#     r"""
#     This is the configuration class to store the configuration of a [`RobertaModel`] or a [`TFRobertaModel`]. It is
#     used to instantiate a RoBERTa model according to the specified arguments, defining the model architecture.
#     Instantiating a configuration with the defaults will yield a similar configuration to that of the RoBERTa
#     [roberta-base](https://huggingface.co/roberta-base) architecture.
#
#     Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
#     documentation from [`PretrainedConfig`] for more information.
#
#     The [`RobertaConfig`] class directly inherits [`BertConfig`]. It reuses the same defaults. Please check the parent
#     class for more information.
#
#     Examples:
#
#     ```python
#     >>> from transformers import RobertaConfig, RobertaModel
#
#     >>> # Initializing a RoBERTa configuration
#     >>> configuration = RobertaConfig()
#
#     >>> # Initializing a model from the configuration
#     >>> model = RobertaModel(configuration)
#
#     >>> # Accessing the model configuration
#     >>> configuration = model.config
#     ```"""
#     model_type = "roberta_pkgm"
#
#     def __init__(self, num_entities, num_relations, kg_embedding_dim, max_seq_len, max_pvs, **kwargs):
#         """Constructs RobertaConfig."""
#         super().__init__(num_entities=num_entities, num_relations=num_relations,
#                          kg_embedding_dim=kg_embedding_dim, max_seq_len=max_seq_len,
#                          max_pvs=max_pvs, **kwargs)
#


class AuxiliaryTaskPair(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(AuxiliaryTaskPair, self).__init__()
        self.config = config
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size * 2, config.num_labels)

    def forward(self, sequence_output, pair_indices):
        x, y, labels = [], [], []
        for i, pair_index in enumerate(pair_indices):
            for j in range(pair_index.size(0)):
                x.append(sequence_output[i, pair_index[j][0]:pair_index[j][1], :].mean(axis=0))
                y.append(sequence_output[i, pair_index[j][2]:pair_index[j][3], :].mean(axis=0))
                labels.append(pair_index[j][4])
        x = torch.stack(x)
        y = torch.stack(y)
        labels = torch.stack(labels)

        x = self.dropout(x)
        # x = self.dense(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)

        y = self.dropout(y)
        # y = self.dense(y)
        # y = torch.tanh(y)
        # y = self.dropout(y)

        logits = self.out_proj(torch.cat((x, y), dim=1))

        return logits, labels


class RobertaPKGMPooler(nn.Module):
    ''' Because input sequence consists of 4 parts:
     [CLS] src_text_ids [SEP] src_kg_ids [CLS] tgt_text_ids [SEP] tgt_kg_ids

    Therefore choose the 2 [CLS] tokens as representation of two sequences respectively
    '''
    def __init__(self, config):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.max_pvs = config.max_pvs
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cls_indices = torch.tensor([0, self.max_seq_len+self.max_pvs+1], dtype=torch.int)
        first_tokens_tensor = hidden_states.index_select(dim=1,
                                                         index=cls_indices)
        pooled_output = self.dense(first_tokens_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# PKGM
class RobertaPKGMModel(RobertaPreTrainedModel):
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

    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.config = config
        self.embeddings = RobertaPKGMEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.pooler = RobertaPKGMPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_text_embeddings(self):
        return self.tet_embeddings.word_embeddings

    def set_text_embeddings(self, value):
        self.text_embeddings.word_embeddings = value

    def get_embeddings(self):
        # TODO: to be implemented
        pass

    def set_entity_embeddings(self, value, projection_matrix):
        self.entity_embeddings = value
        self.entity_projection = projection_matrix

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
            position_ids: torch.Tensor,
            inputs_embeds: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
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
        if input_ids is None or attention_mask is None or token_type_ids is None or position_ids is None:
            raise ValueError("You have to specify input_ids, attention_mask, token_type_ids and position_ids")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        input_shape = token_type_ids.shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

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

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

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


class PKGMTwoTower(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaPKGMModel(config, add_pooling_layer=False)
        # self.classifier = VecSimClassificationHead(config)
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
            input_ids_2: Optional[torch.LongTensor] = None,
            attention_mask_2: Optional[torch.FloatTensor] = None,
            token_type_ids_2: Optional[torch.LongTensor] = None,
            position_ids_2: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
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

        outputs_1 = self.roberta(
            input_ids_1,
            attention_mask=attention_mask_1,
            token_type_ids=token_type_ids_1,
            position_ids=position_ids_1,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output_1 = outputs_1[0]

        outputs_2 = self.roberta(
            input_ids_2,
            attention_mask=attention_mask_2,
            token_type_ids=token_type_ids_2,
            position_ids=position_ids_2,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output_2 = outputs_2[0]

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

        return SequenceClassifierOutput(
            loss=loss,
            probs=probs,
            logits=logits,
            src_embeds=src_embeds,
            tgt_embeds=tgt_embeds,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                    - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                      `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
                      `True`.
                    - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments `config` and `state_dict`).
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):
                Can be either:

                    - an instance of a class derived from [`PretrainedConfig`],
                    - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            state_dict (`Dict[str, torch.Tensor]`, *optional*):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
                [`~PreTrainedModel.from_pretrained`] is not a simpler option.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_tf (`bool`, *optional*, defaults to `False`):
                Load the model weights from a TensorFlow checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            from_flax (`bool`, *optional*, defaults to `False`):
                Load the model weights from a Flax checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `transformers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            _fast_init(`bool`, *optional*, defaults to `True`):
                Whether or not to disable fast initialization.

                <Tip warning={true}>

                One should only disable *_fast_init* to ensure backwards compatibility with `transformers.__version__ <
                4.6.0` for seeded model initialization. This argument will be removed at the next major version. See
                [pull request 11471](https://github.com/huggingface/transformers/pull/11471) for more information.

                </Tip>

            low_cpu_mem_usage(`bool`, *optional*, defaults to `False`):
                Tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                This is an experimental feature and a subject to change at any moment.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype. If `"auto"` is passed the dtype
                will be automatically derived from the model's weights.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
                      corresponds to a configuration attribute will be used to override said attribute with the
                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                      will be passed to the underlying model's `__init__` function.

        <Tip>

        Passing `use_auth_token=True`` is required when you want to use a private model.

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
        use this method in a firewalled environment.

        </Tip>

        Examples:

        ```python
        >>> from transformers import BertConfig, BertModel

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = BertModel.from_pretrained("bert-base-uncased")
        >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
        >>> model = BertModel.from_pretrained("./test/saved_model/")
        >>> # Update configuration during loading.
        >>> model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
        >>> assert model.config.output_attentions == True
        >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
        >>> config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")
        >>> model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)
        >>> # Loading from a Flax checkpoint file instead of a PyTorch model (slower)
        >>> model = BertModel.from_pretrained("bert-base-uncased", from_flax=True)
        ```

        * `low_cpu_mem_usage` algorithm:

        This is an experimental function that loads the model using ~1x model size CPU memory

        Here is how it works:

        1. save which state_dict keys we have
        2. drop state_dict before the model is created, since the latter takes 1x model size CPU memory
        3. after the model has been instantiated switch to the meta device all params/buffers that
        are going to be replaced from the loaded state_dict
        4. load state_dict 2nd time
        5. replace the params/buffers from the state_dict

        Currently, it can't handle deepspeed ZeRO stage 3 and ignores loading errors

        """
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        mirror = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)

        # from_pt = not (from_tf | from_flax)

        # user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        # if from_pipeline is not None:
        #     user_agent["using_pipeline"] = from_pipeline
        #
        # if is_offline_mode() and not local_files_only:
        #     logger.info("Offline mode: forcing local_files_only=True")
        #     local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False
        sharded_metadata = None
        # Load model
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            # roberta model path
            roberta_archive_file = os.path.join(pretrained_model_name_or_path, ROBERTA_WEIGHTS_NAME)
            roberta_resolved_archive_file = roberta_archive_file
            # kg model path
            kg_archive_file = os.path.join(pretrained_model_name_or_path, KG_WEIGHTS_NAME)
            kg_resolved_archive_file = kg_archive_file
            #
            # if resolved_archive_file == archive_file:
            #     logger.info(f"loading weights file {archive_file}")
            # else:
            #     logger.info(f"loading weights file {archive_file} from cache at {resolved_archive_file}")
        else:
            roberta_resolved_archive_file = None
            kg_resolved_archive_file = None

        loaded_state_dict_keys = None
        if not is_sharded and state_dict is None:
            # load roberta checkpoint
            roberta_state_dict = torch.load(roberta_resolved_archive_file, map_location="cpu")
            # load kg checkpoint
            kg_state_dict = torch.load(kg_resolved_archive_file, map_location="cpu")
            # merge state dict
            state_dict = []
            for k, v in roberta_state_dict.items():
                # state_dict.append((k.replace("bert.", ""), v))
                state_dict.append((k, v))
            for k, v in kg_state_dict.items():
                # state_dict.append(("embeddings." + k, v))
                state_dict.append((k, v))
            state_dict = OrderedDict(state_dict)
            # del roberta_state_dict
            # del kg_state_dict
            loaded_state_dict_keys = [k for k in state_dict.keys()]

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
            model,
            state_dict,
            loaded_state_dict_keys,  # XXX: rename?
            roberta_resolved_archive_file,
            pretrained_model_name_or_path,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            sharded_metadata=sharded_metadata,
            _fast_init=_fast_init,
            low_cpu_mem_usage=low_cpu_mem_usage
        )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        return model


class PKGMOneTower(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaPKGMModel(config, add_pooling_layer=False)
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
            inputs_embeds: Optional[torch.FloatTensor] = None,
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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        if self.config.classification_method == "vec_sim":
            src_sequence_output = sequence_output[:, 0, :]
            tgt_sequence_output = sequence_output[:, self.config.max_seq_len+2*self.config.max_pvs, :]
            src_embeds, tgt_embeds, logits, probs = self.classifier(src_sequence_output, tgt_sequence_output)
        else:
            logits = self.classifier(sequence_output)
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

        return SequenceClassifierOutput(
            loss=loss,
            probs=probs,
            logits=logits,
            src_embeds=src_embeds,
            tgt_embeds=tgt_embeds,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                    - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                      `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
                      `True`.
                    - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments `config` and `state_dict`).
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):
                Can be either:

                    - an instance of a class derived from [`PretrainedConfig`],
                    - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            state_dict (`Dict[str, torch.Tensor]`, *optional*):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
                [`~PreTrainedModel.from_pretrained`] is not a simpler option.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_tf (`bool`, *optional*, defaults to `False`):
                Load the model weights from a TensorFlow checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            from_flax (`bool`, *optional*, defaults to `False`):
                Load the model weights from a Flax checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `transformers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            _fast_init(`bool`, *optional*, defaults to `True`):
                Whether or not to disable fast initialization.

                <Tip warning={true}>

                One should only disable *_fast_init* to ensure backwards compatibility with `transformers.__version__ <
                4.6.0` for seeded model initialization. This argument will be removed at the next major version. See
                [pull request 11471](https://github.com/huggingface/transformers/pull/11471) for more information.

                </Tip>

            low_cpu_mem_usage(`bool`, *optional*, defaults to `False`):
                Tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                This is an experimental feature and a subject to change at any moment.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype. If `"auto"` is passed the dtype
                will be automatically derived from the model's weights.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
                      corresponds to a configuration attribute will be used to override said attribute with the
                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                      will be passed to the underlying model's `__init__` function.

        <Tip>

        Passing `use_auth_token=True`` is required when you want to use a private model.

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
        use this method in a firewalled environment.

        </Tip>

        Examples:

        ```python
        >>> from transformers import BertConfig, BertModel

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = BertModel.from_pretrained("bert-base-uncased")
        >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
        >>> model = BertModel.from_pretrained("./test/saved_model/")
        >>> # Update configuration during loading.
        >>> model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
        >>> assert model.config.output_attentions == True
        >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
        >>> config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")
        >>> model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)
        >>> # Loading from a Flax checkpoint file instead of a PyTorch model (slower)
        >>> model = BertModel.from_pretrained("bert-base-uncased", from_flax=True)
        ```

        * `low_cpu_mem_usage` algorithm:

        This is an experimental function that loads the model using ~1x model size CPU memory

        Here is how it works:

        1. save which state_dict keys we have
        2. drop state_dict before the model is created, since the latter takes 1x model size CPU memory
        3. after the model has been instantiated switch to the meta device all params/buffers that
        are going to be replaced from the loaded state_dict
        4. load state_dict 2nd time
        5. replace the params/buffers from the state_dict

        Currently, it can't handle deepspeed ZeRO stage 3 and ignores loading errors

        """
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        mirror = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)

        # from_pt = not (from_tf | from_flax)

        # user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        # if from_pipeline is not None:
        #     user_agent["using_pipeline"] = from_pipeline
        #
        # if is_offline_mode() and not local_files_only:
        #     logger.info("Offline mode: forcing local_files_only=True")
        #     local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False
        sharded_metadata = None
        # Load model
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            # roberta model path
            roberta_archive_file = os.path.join(pretrained_model_name_or_path, ROBERTA_WEIGHTS_NAME)
            roberta_resolved_archive_file = roberta_archive_file
            # kg model path
            kg_archive_file = os.path.join(pretrained_model_name_or_path, KG_WEIGHTS_NAME)
            kg_resolved_archive_file = kg_archive_file
            #
            # if resolved_archive_file == archive_file:
            #     logger.info(f"loading weights file {archive_file}")
            # else:
            #     logger.info(f"loading weights file {archive_file} from cache at {resolved_archive_file}")
        else:
            roberta_resolved_archive_file = None
            kg_resolved_archive_file = None

        loaded_state_dict_keys = None
        if not is_sharded and state_dict is None:
            # load roberta checkpoint
            roberta_state_dict = torch.load(roberta_resolved_archive_file, map_location="cpu")
            # load kg checkpoint
            kg_state_dict = torch.load(kg_resolved_archive_file, map_location="cpu")
            # merge state dict
            state_dict = []
            for k, v in roberta_state_dict.items():
                # state_dict.append((k.replace("bert.", ""), v))
                state_dict.append((k, v))
            for k, v in kg_state_dict.items():
                # state_dict.append(("embeddings." + k, v))
                state_dict.append((k, v))
            state_dict = OrderedDict(state_dict)
            # del roberta_state_dict
            # del kg_state_dict
            loaded_state_dict_keys = [k for k in state_dict.keys()]

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
            model,
            state_dict,
            loaded_state_dict_keys,  # XXX: rename?
            roberta_resolved_archive_file,
            pretrained_model_name_or_path,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            sharded_metadata=sharded_metadata,
            _fast_init=_fast_init,
            low_cpu_mem_usage=low_cpu_mem_usage
        )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        return model


# Roberta
class RobertaModel(RobertaPreTrainedModel):
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

    # @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=BaseModelOutputWithPoolingAndCrossAttentions,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    # # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            cate_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
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

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
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

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            cate_ids=cate_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
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


class RobertaTwoTower(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # self.classifier = VecSimClassificationHead(config)
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
            cate_ids_1: Optional[torch.LongTensor] = None,
            position_ids_1: Optional[torch.LongTensor] = None,
            input_ids_2: Optional[torch.LongTensor] = None,
            attention_mask_2: Optional[torch.FloatTensor] = None,
            token_type_ids_2: Optional[torch.LongTensor] = None,
            cate_ids_2: Optional[torch.LongTensor] = None,
            position_ids_2: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
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

        outputs_1 = self.roberta(
            input_ids_1,
            attention_mask=attention_mask_1,
            token_type_ids=token_type_ids_1,
            position_ids=position_ids_1,
            cate_ids=cate_ids_1,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output_1 = outputs_1[0]

        outputs_2 = self.roberta(
            input_ids_2,
            attention_mask=attention_mask_2,
            token_type_ids=token_type_ids_2,
            cate_ids=cate_ids_2,
            position_ids=position_ids_2,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output_2 = outputs_2[0]

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
            probs=probs,
            logits=logits,
            src_embeds=src_embeds,
            tgt_embeds=tgt_embeds
        )


class RobertaOneTower(RobertaPreTrainedModel):
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

        self.roberta = RobertaModel(config, add_pooling_layer=False)
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
        if config.auxiliary_task:
            self.auxiliary_task = AuxiliaryTaskPair(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            cate_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
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
            cate_ids=cate_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
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
            logits = self.classifier(sequence_output)
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
            if self.config.auxiliary_task:
                logits2, labels2 = self.auxiliary_task(sequence_output, image_indices)
                loss += self.loss_fct(logits2.view(-1, self.num_labels), labels2.view(-1))

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


# TextCNN
class TextCNN(nn.Module):

    def __init__(self, config, embedding_state_dict):
        super(TextCNN, self).__init__()
        filter_sizes = [int(i) for i in config.filter_sizes.split(",")] # [1, 2, 3, 5]
        # non-static channel
        self.embedding1 = RobertaEmbeddings(config)
        self.embedding1.load_state_dict(embedding_state_dict, strict=False)
        # static channel
        self.embedding2 = RobertaEmbeddings(config)
        self.embedding2.load_state_dict(embedding_state_dict, strict=False)
        for key, value in dict(self.embedding2.named_parameters()).items():
            value.requires_grad = False
        # convolutional layers
        self.convs1 = nn.ModuleList([nn.Conv2d(2, config.num_filters, (K, config.hidden_size)) for K in filter_sizes])
        # dropout layer
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # feedforward layer
        # self.fc1 = nn.Linear(len(filter_sizes)*config.num_filters, config.hidden_size)

    def forward(self, x):
        x1 = self.embedding1(x)
        x2 = self.embedding2(x)
        x = torch.stack((x1, x2), dim=1)
        # x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        # x = self.fc1(x)

        return x


class TextCNNTwoTower(nn.Module):

    def __init__(self, config, embedding_state_dict):
        super(TextCNNTwoTower, self).__init__()
        self.num_labels = config.num_labels
        self.config = config

        self.textcnn = TextCNN(config, embedding_state_dict)
        if config.classification_method == "vec_sim":
            self.classifier = VecSimClassificationHead(config)
        else:
            hidden_size = len(config.filter_sizes.split(",")) * config.num_filters
            self.classifier = TwoTowerClassificationHead(hidden_size, config.hidden_dropout_prob)

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
            input_ids_1: Optional[torch.LongTensor] = None,
            attention_mask_1: Optional[torch.FloatTensor] = None,
            token_type_ids_1: Optional[torch.LongTensor] = None,
            position_ids_1: Optional[torch.LongTensor] = None,
            input_ids_2: Optional[torch.LongTensor] = None,
            attention_mask_2: Optional[torch.FloatTensor] = None,
            token_type_ids_2: Optional[torch.LongTensor] = None,
            position_ids_2: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
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

        outputs_1 = self.textcnn(input_ids_1)

        outputs_2 = self.textcnn(input_ids_2)

        src_embeds, tgt_embeds, logits, probs = self.classifier(outputs_1, outputs_2)
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
