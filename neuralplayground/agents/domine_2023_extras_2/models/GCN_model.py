import torch.nn as nn
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
# TODO(clementine): set up object oriented GNN classes (eventually)

class GCNModel(nn.Module):
    def __init__(self, num_hidden, num_layers, num_message_passing_steps, residual, layer_norm):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers
        num_features = 3
        self.num_message_passing_steps = num_message_passing_steps
        self.conv_1 = GCNConv(3, 6)
        self.conv_layers = nn.ModuleList([GCNConv(6, 6) for _ in range(num_message_passing_steps)])
        self.fc = nn.Linear(6, 3)  # Output layer
        self.residual = residual
        self.layer_norm = layer_norm
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(num_hidden) for _ in range(num_message_passing_steps)]) if layer_norm else None

    def forward(self, node,edges):
        x, edge_index = node, edges
        x = self.conv_1(x, edge_index)
        for i, conv in enumerate(self.conv_layers):
            x_res = x
            x = conv(x, edge_index)
            if self.layer_norm:
                x = self.norm_layers[i](x)
            if self.residual:
                x += x_res
            x = torch.relu(x)

        x = self.fc(x)
        return x