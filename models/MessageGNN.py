from typing import Optional
import torch
from torch import nn, Tensor
from torch_geometric.nn import MessagePassing, GraphNorm
from torch_geometric.nn.aggr import SumAggregation
import logging
import copy

logger = logging.getLogger(__name__)


class GraphLayer(MessagePassing):
    def __init__(self, embedding_size): # Verify along which axis to propagate
        super().__init__(aggr=None)

        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size),
            nn.Tanh()
        )

        self.mlp_upd = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size),
            nn.Tanh()
        )

        self.aggr_m = SumAggregation()

        #self.norm = GraphNorm(out_channels)


    def forward(self,
                node_feature: Tensor,
                edge_index: Tensor,
                edge_feature: Tensor,
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
                              edge_feature=edge_feature,
                              batch=batch)

    def aggregate(self,
                  mes: Tensor,
                  index: Tensor) -> Tensor:

        aggregated = self.aggr_m(x=mes,
                                 index=index)

        return aggregated, mes


    def message(self, node_feature_i, node_feature_j, edge_feature, batch):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        """
        Constructs messages to node i
        :param edge_feature:
        :param node_feature_j:
        :param node_feature_i:
        :return:
        """
        # Use below if there are edge attributes
        #m = torch.cat([node_feature_i, node_feature_j, edge_feature[:,0:1]], dim=1)
        m = torch.cat((node_feature_j, edge_feature), dim=1)
        mes = self.mlp_msg(m)

        return mes

    def update(self, aggr, node_feature, batch) -> Tensor:
        aggr_msg = aggr[0]
        edge_upd = aggr[1]

        msg_to_upd = torch.cat((node_feature, aggr_msg), dim=1)
        node_feature_out = self.mlp_upd(msg_to_upd)

        return node_feature_out, edge_upd


class MessagePassingGNN(nn.Module):
    def __init__(self,
                 embedding_size: int,
                 layers: int,
                 input_size = 2):
        super().__init__()

        self.embedding_size = embedding_size

        self.embedding = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.Tanh()
        )

        graph_layers = [GraphLayer(embedding_size) for _ in range(layers)]
        self.graph_layers = nn.ModuleList(graph_layers)

        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, embedding_size// 2),
            nn.Tanh(),
            nn.Linear(embedding_size // 2,   1)
        )

    def forward(self,
                node_feature: Tensor,
                edge_index: Tensor,
                edge_feature: Tensor,
                batch: Tensor):

        emb_edge_feature = self.embedding(edge_feature)

        # do embedding for edge attributes
        for layer in self.graph_layers:
            # edge embeddings and node embeddings are updated
            node_feature, emb_edge_feature = layer(node_feature, edge_index, emb_edge_feature, batch)

        out = self.decoder(node_feature)

        return out