
import torch

from torch import nn
import torch.nn.functional as F
from .base import SequenceClassifierOutput, TwoTowerClassificationHead
from torch_geometric.nn import GCN2Conv
from ..utils import logger


# GCN
class GCN(torch.nn.Module):
    def __init__(self, config):
        # intermediate_size, num_hidden_layers, hidden_dropout_prob=0.1,
        #          hidden_size=768, alpha=0.1, theta=0.5, num_labels=2):
        super().__init__()

        self.linear = torch.nn.Linear(config.hidden_size, config.intermediate_size)
        # self.lins = torch.nn.ModuleList()
        # self.lins.append(torch.nn.Linear(hidden_size, intermediate_size))
        # self.lins.append(torch.nn.Linear(intermediate_size*2, num_labels))

        self.convs = torch.nn.ModuleList()
        for layer in range(config.num_hidden_layers):
            self.convs.append(
                GCN2Conv(config.intermediate_size, config.alpha, config.theta, layer + 1,
                         shared_weights=True, normalize=False))

        self.dropout = config.hidden_dropout_prob

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.linear(x).relu()
        # x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        #         x = self.lins[1](x)

        return x # x.log_softmax(dim=-1)


class GCNTwoTower(nn.Module):
    def __init__(self, config):
    #         intermediate_size,
    #         num_hidden_layers,
    #         hidden_dropout_prob=0.1,
    #         hidden_size=768,
    #         num_labels=2,
    #         loss_margin=0.0
    # ):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        # GCN encoder
        self.encoder = GCN(config)
                           # intermediate_size,
                           # num_hidden_layers,
                           # hidden_dropout_prob=hidden_dropout_prob,
                           # hidden_size=hidden_size)

        # self.classifier = RobertaClassificationHead(config)
        self.classifier = TwoTowerClassificationHead(config.intermediate_size,
                                                     dropout=config.hidden_dropout_prob,
                                                     num_labels=config.num_labels)

        # if config.loss_type == "cosine":
        #     self.loss_fct = nn.CosineEmbeddingLoss(margin=loss_margin)
        # elif config.loss_type == "bce":
        #     self.loss_fct = nn.BCEWithLogitsLoss()
        # elif config.loss_type == "euclidean":
        #     self.loss_fct = EuclideanDistanceLoss()
        # elif config.loss_type == "hinge":
        #     self.loss_fct = HingeLoss(margin=loss_margin)
        # else:
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, feature_matrix, adjacency_matrix, pairs):
        node_embeddings = self.encoder(feature_matrix, adjacency_matrix)
        loss = None
        logits, probs, src_embeds, tgt_embeds = None, None, None, None
        for pair in pairs:
            i = pair['src_idx']
            j = pair['tgt_idx']
            src_node_embeddings = node_embeddings[i].unsqueeze(0)
            tgt_node_embeddings = node_embeddings[j].unsqueeze(0)

            # logits
            src_e, tgt_e, lgt, prob = self.classifier(src_node_embeddings, tgt_node_embeddings)
            if logits is None:
                logits = lgt
                probs = prob[:, 1]
                src_embeds = prob[:, 0]
                tgt_embeds = prob[:, 1]
            else:
                src_embeds = torch.cat((src_embeds, prob[:, 0]), dim=0)
                tgt_embeds = torch.cat((tgt_embeds, prob[:, 1]), dim=0)
                probs = torch.cat((probs, prob[:, 1]), dim=0)

            labels = pair.get('item_label', None)
            if labels is not None:
                if loss is None:
                    loss = 0
                labels = torch.tensor(int(pair['item_label']), dtype=torch.long, device=feature_matrix.device)
                if self.config.loss_type == "cosine":
                    loss += self.loss_fct(src_embeds, tgt_embeds, (labels*2-1).view(-1))
                elif self.config.loss_type == "ce":
                    loss += self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.loss_type == "hinge" or self.config.loss_type == "euclidean":
                    loss += self.loss_fct(logits.view(-1), (labels*2-1).view(-1))
                else:
                    loss += self.loss_fct(logits.view(-1), labels.view(-1))

        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        if loss is not None:
            loss /= len(pairs)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            probs=probs,
            src_embeds=src_embeds,
            tgt_embeds=tgt_embeds
        )
