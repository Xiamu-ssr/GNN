import torch_geometric.nn
import torch.nn.functional as F
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, num_out_heads):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads,
                            dropout=0.6)  
        self.gat2 = GATConv(hidden_channels * num_heads, out_channels,
                            heads=num_out_heads, concat=False,
                            dropout=0.6)

    def forward(self, x : Data.x, edge_index : Data.edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)