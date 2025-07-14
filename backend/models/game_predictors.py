import torch
from torch import nn
import torch_geometric 
from torch_geometric import nn as gnn

class GAT(nn.module):
    def __init__(self,
                 gat_input_size,
                 gat_num_layers,
                 gat_output_size=None
                 ):
        super.__init__()

        self.gnn = gnn.models.GAT(gat_input_size, gat_num_layers, gat_output_size)
        self.softmax = nn.softmax(dim=1)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.gnn(x, edge_index, batch=batch)
        x = gnn.pooling.global_mean_pool(x, batch)
        x = self.softmax(x)
        return x
