import torch.nn as nn
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv,global_mean_pool
# TODO(clementine): set up object oriented GNN classes (eventually)

class GCNModel(nn.Module):
    def __init__(self, num_hidden,num_feature, num_layers, num_message_passing_steps, residual, layer_norm):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers
        self.num_message_passing_steps = num_message_passing_steps
        self.conv_1 = GCNConv(num_feature, num_hidden)
        self.conv_layers = nn.ModuleList([GCNConv(num_hidden, num_hidden) for _ in range(num_message_passing_steps)])
        self.fc = nn.Linear(num_hidden,1)  # Output layer
        self.residual = residual
        self.layer_norm = layer_norm
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(num_hidden) for _ in range(num_message_passing_steps)]) if layer_norm else None

    def forward(self, node,edges,edges_attr):
        x, edge_index = node, edges
        x = self.conv_1(x, edge_index,edges_attr)
        for i, conv in enumerate(self.conv_layers):
            x_res = x
            x = conv(x, edge_index,edges_attr)
            if self.layer_norm:
                x = self.norm_layers[i](x)
            if self.residual:
                x += x_res
            x = torch.relu(x)

        x = self.fc(x)
        x = global_mean_pool(x, batch=None)
        x = x.view(-1)
        #Flatten the tensor to 1D if necessary #TODO: ask here if this makes sense

        return x