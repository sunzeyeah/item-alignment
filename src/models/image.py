
import torch

from functools import partial
from torch import nn
from timm.models.nfnet import NfCfg, _nonlin_gamma, act_with_gamma, create_stem, NormFreeBlock
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import ScaledStdConv2d, ScaledStdConv2dSame, PatchEmbed, get_act_layer, get_attn, make_divisible
from timm.models.layers.classifier import _create_pool, _create_fc
from .base import SequenceClassifierOutput, TwoTowerClassificationHead
from .loss import HingeLoss, EuclideanDistanceLoss
from ..utils import logger


# eca nfnet
class ClassifierHead(nn.Module):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(self, in_chs, num_classes, pool_type='avg', drop_rate=0., use_conv=False):
        super(ClassifierHead, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool, num_pooled_features = _create_pool(in_chs, num_classes, pool_type, use_conv=use_conv)
        self.fc = _create_fc(num_pooled_features*2, num_classes, use_conv=use_conv)
        # self.flatten = nn.Flatten(1) if use_conv and pool_type else nn.Identity()

    def pool(self, x):
        return self.global_pool(x)

    def forward(self, x, y):
        x = self.pool(x)
        y = self.pool(y)
        if self.drop_rate:
            x = nn.functional.dropout(x, p=float(self.drop_rate), training=self.training)
            y = nn.functional.dropout(y, p=float(self.drop_rate), training=self.training)
        x = self.fc(torch.cat((x, y), dim=-1))
        # x = self.flatten(x)
        return x


class NormFreeNet(nn.Module):
    """ Normalization-Free Network

    As described in :
    `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
        - https://arxiv.org/abs/2101.08692
    and
    `High-Performance Large-Scale Image Recognition Without Normalization` - https://arxiv.org/abs/2102.06171

    This model aims to cover both the NFRegNet-Bx models as detailed in the paper's code snippets and
    the (preact) ResNet models described earlier in the paper.

    There are a few differences:
        * channels are rounded to be divisible by 8 by default (keep tensor core kernels happy),
            this changes channel dim and param counts slightly from the paper models
        * activation correcting gamma constants are moved into the ScaledStdConv as it has less performance
            impact in PyTorch when done with the weight scaling there. This likely wasn't a concern in the JAX impl.
        * a config option `gamma_in_act` can be enabled to not apply gamma in StdConv as described above, but
            apply it in each activation. This is slightly slower, numerically different, but matches official impl.
        * skipinit is disabled by default, it seems to have a rather drastic impact on GPU memory use and throughput
            for what it is/does. Approx 8-10% throughput loss.
    """
    def __init__(
            self, cfg: NfCfg, num_classes=1000, in_chans=3, global_pool='avg', output_stride=32,
            drop_rate=0., drop_path_rate=0.
    ):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        assert cfg.act_layer in _nonlin_gamma, f"Please add non-linearity constants for activation ({cfg.act_layer})."
        conv_layer = ScaledStdConv2dSame if cfg.same_padding else ScaledStdConv2d
        if cfg.gamma_in_act:
            act_layer = act_with_gamma(cfg.act_layer, gamma=_nonlin_gamma[cfg.act_layer])
            conv_layer = partial(conv_layer, eps=cfg.std_conv_eps)
        else:
            act_layer = get_act_layer(cfg.act_layer)
            conv_layer = partial(conv_layer, gamma=_nonlin_gamma[cfg.act_layer], eps=cfg.std_conv_eps)
        attn_layer = partial(get_attn(cfg.attn_layer), **cfg.attn_kwargs) if cfg.attn_layer else None

        stem_chs = make_divisible((cfg.stem_chs or cfg.channels[0]) * cfg.width_factor, cfg.ch_div)
        self.stem, stem_stride, stem_feat = create_stem(
            in_chans, stem_chs, cfg.stem_type, conv_layer=conv_layer, act_layer=act_layer)

        self.feature_info = [stem_feat]
        drop_path_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg.depths)).split(cfg.depths)]
        prev_chs = stem_chs
        net_stride = stem_stride
        dilation = 1
        expected_var = 1.0
        stages = []
        for stage_idx, stage_depth in enumerate(cfg.depths):
            stride = 1 if stage_idx == 0 and stem_stride > 2 else 2
            if net_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            net_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2

            blocks = []
            for block_idx in range(cfg.depths[stage_idx]):
                first_block = block_idx == 0 and stage_idx == 0
                out_chs = make_divisible(cfg.channels[stage_idx] * cfg.width_factor, cfg.ch_div)
                blocks += [NormFreeBlock(
                    in_chs=prev_chs, out_chs=out_chs,
                    alpha=cfg.alpha,
                    beta=1. / expected_var ** 0.5,
                    stride=stride if block_idx == 0 else 1,
                    dilation=dilation,
                    first_dilation=first_dilation,
                    group_size=cfg.group_size,
                    bottle_ratio=1. if cfg.reg and first_block else cfg.bottle_ratio,
                    ch_div=cfg.ch_div,
                    reg=cfg.reg,
                    extra_conv=cfg.extra_conv,
                    skipinit=cfg.skipinit,
                    attn_layer=attn_layer,
                    attn_gain=cfg.attn_gain,
                    act_layer=act_layer,
                    conv_layer=conv_layer,
                    drop_path_rate=drop_path_rates[stage_idx][block_idx],
                )]
                if block_idx == 0:
                    expected_var = 1.  # expected var is reset after first block of each stage
                expected_var += cfg.alpha ** 2   # Even if reset occurs, increment expected variance
                first_dilation = dilation
                prev_chs = out_chs
            self.feature_info += [dict(num_chs=prev_chs, reduction=net_stride, module=f'stages.{stage_idx}')]
            stages += [nn.Sequential(*blocks)]
        self.stages = nn.Sequential(*stages)

        if cfg.num_features:
            # The paper NFRegNet models have an EfficientNet-like final head convolution.
            self.num_features = make_divisible(cfg.width_factor * cfg.num_features, cfg.ch_div)
            self.final_conv = conv_layer(prev_chs, self.num_features, 1)
            self.feature_info[-1] = dict(num_chs=self.num_features, reduction=net_stride, module=f'final_conv')
        else:
            self.num_features = prev_chs
            self.final_conv = nn.Identity()
        self.final_act = act_layer(inplace=cfg.num_features > 0)

        self.classifier = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

        self.loss = nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax()

        for n, m in self.named_modules():
            if 'fc' in n and isinstance(m, nn.Linear):
                if cfg.zero_init_fc:
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.normal_(m.weight, 0., .01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',
            blocks=[
                (r'^stages\.(\d+)' if coarse else r'^stages\.(\d+)\.(\d+)', None),
                (r'^final_conv', (99999,))
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    # @torch.jit.ignore
    # def get_classifier(self):
    #     return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        x = self.final_conv(x)
        x = self.final_act(x)
        return x

    def forward(self, x, y, labels):
        x = self.forward_features(x)
        y = self.forward_features(y)
        logits = self.classifier(x, y)
        probs = self.softmax(logits)
        src_embeds = probs[:, 0]
        tgt_embeds = probs[:, 1]
        probs = probs[:, 1]

        loss = None
        if labels is not None:
            loss = self.loss(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            probs=probs,
            src_embeds=src_embeds,
            tgt_embeds=tgt_embeds
        )


class NFNetTwoTower(nn.Module):
    def __init__(
            self,
            config,
            image_encoder
    ):
        super().__init__()
        self.config = config
        # self.dim = config.hidden_size
        self.num_labels = config.num_labels

        # image encoder
        self.img_encoder = image_encoder

        # # attention pooling for image tokens
        # num_image_queries = config.num_image_queries + 1
        # self.img_queries = nn.Parameter(torch.randn(num_image_queries, config.hidden_size)) # num image queries for multimodal, but 1 extra CLS for contrastive learning
        # hidden_size_head = config.hidden_size // config.num_attention_heads
        # self.img_attn_pool = CrossAttention(dim=config.hidden_size, context_dim=config.hidden_size,
        #                                     dim_head=hidden_size_head, heads=config.num_attention_heads,
        #                                     norm_context=True)
        # self.img_attn_pool_norm = LayerNorm(config.hidden_size)

        # self.classifier = RobertaClassificationHead(config)
        self.classifier = TwoTowerClassificationHead(image_encoder.num_features,
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

    def forward(self, images_1, images_2, labels):
        image_tokens_1 = self.img_encoder.forward_features(images_1)
        image_tokens_1 = self.img_encoder.head.global_pool(image_tokens_1)
        image_tokens_2 = self.img_encoder.forward_features(images_2)
        image_tokens_2 = self.img_encoder.head.global_pool(image_tokens_2)

        # # attention pool image tokens
        # img_queries = repeat(self.img_queries, 'n d -> b n d', b=image_tokens_1.shape[0])
        # img_queries_1 = self.img_attn_pool(img_queries, image_tokens_1)
        # sequence_output_1 = self.img_attn_pool_norm(img_queries_1)
        #
        # img_queries_2 = self.img_attn_pool(img_queries, image_tokens_2)
        # sequence_output_2 = self.img_attn_pool_norm(img_queries_2)

        # logits
        src_embeds, tgt_embeds, logits, probs = self.classifier(image_tokens_1, image_tokens_2)
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
            logits=logits,
            probs=probs,
            src_embeds=src_embeds,
            tgt_embeds=tgt_embeds
        )


#ResNet
class ResNetTwoTower(nn.Module):
    def __init__(
            self,
            config,
            image_encoder
    ):
        super().__init__()
        self.config = config
        # self.dim = config.hidden_size
        self.num_labels = config.num_labels

        # image encoder
        self.img_encoder = image_encoder

        # # attention pooling for image tokens
        # num_image_queries = config.num_image_queries + 1
        # self.img_queries = nn.Parameter(torch.randn(num_image_queries, config.hidden_size)) # num image queries for multimodal, but 1 extra CLS for contrastive learning
        # hidden_size_head = config.hidden_size // config.num_attention_heads
        # self.img_attn_pool = CrossAttention(dim=config.hidden_size, context_dim=config.hidden_size,
        #                                     dim_head=hidden_size_head, heads=config.num_attention_heads,
        #                                     norm_context=True)
        # self.img_attn_pool_norm = LayerNorm(config.hidden_size)

        # self.classifier = RobertaClassificationHead(config)
        self.classifier = TwoTowerClassificationHead(image_encoder.num_features,
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

    def forward(self, images_1, images_2, labels):
        image_tokens_1 = self.img_encoder.forward_features(images_1)
        image_tokens_1 = self.img_encoder.head.global_pool(image_tokens_1).flatten(1)
        image_tokens_2 = self.img_encoder.forward_features(images_2)
        image_tokens_2 = self.img_encoder.head.global_pool(image_tokens_2).flatten(1)

        # # attention pool image tokens
        # img_queries = repeat(self.img_queries, 'n d -> b n d', b=image_tokens_1.shape[0])
        # img_queries_1 = self.img_attn_pool(img_queries, image_tokens_1)
        # sequence_output_1 = self.img_attn_pool_norm(img_queries_1)
        #
        # img_queries_2 = self.img_attn_pool(img_queries, image_tokens_2)
        # sequence_output_2 = self.img_attn_pool_norm(img_queries_2)

        # logits
        src_embeds, tgt_embeds, logits, probs = self.classifier(image_tokens_1, image_tokens_2)
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
            logits=logits,
            probs=probs,
            src_embeds=src_embeds,
            tgt_embeds=tgt_embeds
        )


# ViT
class ViT(VisionTransformer):
    def __init__(self, config):
        super(ViT, self).__init__(img_size=config.image_size,
                                  patch_size=config.patch_size, in_chans=3, num_classes=1000,
                                  embed_dim=config.hidden_size, depth=config.num_hidden_layers,
                                  num_heads=config.num_attention_heads,
                                  mlp_ratio=4.,
                                  qkv_bias=True, representation_size=None, distilled=False,
                                  drop_rate=config.hidden_dropout_prob,
                                  attn_drop_rate=config.attention_probs_dropout_prob,
                                  drop_path_rate=0.,
                                  embed_layer=PatchEmbed,
                                  norm_layer=None,
                                  act_layer=None, weight_init='')

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        # cls_out = None
        # if self.dist_token is None:
        #     cls_out = self.pre_logits(x[:, 0])
        # seq_out = x[:, 1]

        return x[:, 0], x[:, 1:]


class VitTwoTower(nn.Module):
    def __init__(
            self,
            config,
            image_encoder
    ):
        super().__init__()
        self.config = config
        # self.dim = config.hidden_size
        self.num_labels = config.num_labels

        # image encoder
        self.img_encoder = image_encoder

        # # attention pooling for image tokens
        # num_image_queries = config.num_image_queries + 1
        # self.img_queries = nn.Parameter(torch.randn(num_image_queries, config.hidden_size)) # num image queries for multimodal, but 1 extra CLS for contrastive learning
        # hidden_size_head = config.hidden_size // config.num_attention_heads
        # self.img_attn_pool = CrossAttention(dim=config.hidden_size, context_dim=config.hidden_size,
        #                                     dim_head=hidden_size_head, heads=config.num_attention_heads,
        #                                     norm_context=True)
        # self.img_attn_pool_norm = LayerNorm(config.hidden_size)

        # self.classifier = RobertaClassificationHead(config)
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

    def forward(self, images_1, images_2, labels):
        # encode images into embeddings
        # with the img_encoder passed in at init
        # it can also accept precomputed image tokens

        image_tokens_1 = self.img_encoder.forward_features(images_1)
        image_tokens_1 = self.img_encoder.forward_head(image_tokens_1, pre_logits=True)
        image_tokens_2 = self.img_encoder.forward_features(images_2)
        image_tokens_2 = self.img_encoder.forward_head(image_tokens_2, pre_logits=True)

        # # attention pool image tokens
        # img_queries = repeat(self.img_queries, 'n d -> b n d', b=image_tokens_1.shape[0])
        # img_queries_1 = self.img_attn_pool(img_queries, image_tokens_1)
        # sequence_output_1 = self.img_attn_pool_norm(img_queries_1)
        #
        # img_queries_2 = self.img_attn_pool(img_queries, image_tokens_2)
        # sequence_output_2 = self.img_attn_pool_norm(img_queries_2)

        # logits
        src_embeds, tgt_embeds, logits, probs = self.classifier(image_tokens_1, image_tokens_2)
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
            logits=logits,
            probs=probs,
            src_embeds=src_embeds,
            tgt_embeds=tgt_embeds
        )