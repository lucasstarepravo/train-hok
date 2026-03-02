from typing import Optional
import torch
from torch import nn, Tensor
from torch_geometric.nn import MessagePassing, GraphNorm
from torch.nn import LayerNorm
from torch_geometric.nn.aggr import SumAggregation, AttentionalAggregation
import logging
import copy

logger = logging.getLogger(__name__)


def reset_params(layers, activation: str):
    for layer in layers:
        if isinstance(layer, nn.Linear):
            if activation == 'selu':
                nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain(activation))
                nn.init.zeros_(layer.bias)
            elif activation == 'tanh':
                nn.init.kaiming_normal_(layer.weight, nonlinearity=activation)
                nn.init.zeros_(layer.bias)


class GraphLayer(MessagePassing):
    def __init__(self, embedding_size): # Verify along which axis to propagate
        super().__init__(aggr=None)
        #self.dropout = nn.Dropout(0.2)

        self.attention_mlp = nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.Tanh(),

        )
        reset_params(self.attention_mlp, 'tanh')

        self.attention_aggr = AttentionalAggregation(nn.Sequential(
            nn.Linear(embedding_size, embedding_size)),
            self.attention_mlp
            )

        self.mlp_msg = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.Tanh(),

        )

        reset_params(self.mlp_msg, 'tanh')

        self.mlp_upd = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size),
            nn.Tanh(),

        )
        reset_params(self.mlp_upd, 'tanh')

        #self.node_norm = LayerNorm(embedding_size)

        #self.aggr_m = SumAggregation()

        #self.norm = GraphNorm(out_channels)


    def forward(self,
                node_feature: Tensor,
                edge_index: Tensor,
                batch: Tensor) -> Tensor:
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # propagate_type(node_feature: Tensor, edge_index: Tensor, edge_feature: Tensor)
        """
        Takes in the edge indices and all additional data which is needed to construct messages and to update node
        embeddings.
        :param node_feature:
        :param edge_index:
        :param edge_feature:
        :return:
        """

        return self.propagate(edge_index,
                              node_feature=node_feature,
                              batch=batch)

    def aggregate(self,
                  mes: Tensor,
                  index: Tensor) -> Tensor:
        aggregated = self.attention_aggr(x=mes,
                                         index=index)
        #aggregated = self.aggr_m(x=mes,
        #                         index=index)

        return aggregated#, mes


    def message(self, node_feature_i, node_feature_j, batch):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        """
        Constructs messages to node i
        :param node_feature_j:
        :param node_feature_i:
        :return:
        """
        # Use below if there are edge attributes
        #m = torch.cat([node_feature_i, node_feature_j, edge_feature[:,0:1]], dim=1)
        mes = self.mlp_msg(node_feature_j)

        return mes

    def update(self, aggr, node_feature, batch) -> Tensor:
        aggr_msg = aggr
        #edge_upd = self.edge_norm(aggr[1])

        msg_to_upd = torch.cat((node_feature, aggr_msg), dim=1)
        node_feature_out = self.mlp_upd(msg_to_upd)

        return node_feature_out#, edge_upd


class SmallGNN(nn.Module):
    def __init__(self,
                 embedding_size: int,
                 layers: int,
                 input_size: int=2,
                 output_size: int=1):
        super().__init__()

        self.embedding_size = embedding_size

        self.node_encoder  = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.Tanh()
        )
        reset_params(self.node_encoder, 'tanh')

        #self.edge_encoder = nn.Sequential(
        #    nn.Linear(input_size, embedding_size),
        #    nn.Tanh()
        #)
        #reset_params(self.edge_encoder, 'tanh')

        graph_layers = []
        for _ in range(layers):
            graph_layers.append(GraphLayer(embedding_size))

        self.graph_layers = nn.ModuleList(graph_layers)

        self.decoder1 = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.Tanh()
        )
        reset_params(self.decoder1, 'tanh')

        self.decoder2 = nn.Sequential(nn.Linear(embedding_size // 2,  output_size))

    def forward(self,
                node_feature: Tensor,
                edge_index: Tensor,
                batch: Tensor):

        #emb_edge_feature = self.edge_encoder(edge_feature)

        emb_node_feature = self.node_encoder(node_feature)

        # do embedding for edge attributes
        for layer in self.graph_layers:
            # edge embeddings and node embeddings are updated
            emb_node_feature = layer(emb_node_feature, edge_index, batch)


        out = self.decoder1(emb_node_feature)
        out = self.decoder2(out)

        return out