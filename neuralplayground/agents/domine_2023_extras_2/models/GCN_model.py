import torch.nn as nn
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv,global_mean_pool
# TODO(clementine): set up object oriented GNN classes (eventually)
class GCNModel(nn.Module):
    def __init__(self, num_hidden, num_feature, num_layers, num_message_passing_steps, residual, layer_norm):
        super(GCNModel, self).__init__()
        self.num_message_passing_steps = num_message_passing_steps
        self.conv_1 = GCNConv(num_feature, num_hidden)
        self.conv_layers = nn.ModuleList([GCNConv(num_hidden, num_hidden) for _ in range(num_message_passing_steps)])

        # Output layer with size 2 for binary classification logits
        self.fc = nn.Linear(num_hidden, 2)

        self.residual = residual
        self.layer_norm = layer_norm
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(num_hidden) for _ in range(num_message_passing_steps)]) if layer_norm else None

        # Define the softmax layer
        self.softmax = nn.Softmax(dim=1)  # Apply softmax across the logits (along the class dimension)

    def forward(self, node, edges, edges_attr):
        x, edge_index = node, edges
        x = self.conv_1(x, edge_index, edges_attr)

        for i, conv in enumerate(self.conv_layers):
            x_res = x
            x = conv(x, edge_index, edges_attr)
            if self.layer_norm:
                x = self.norm_layers[i](x)
            if self.residual:
                x += x_res
            x = torch.relu(x)

        # The output layer now produces 2 logits for each graph node
        x = self.fc(x)

        # Pooling to get a graph-level representation
        x = global_mean_pool(x, batch=None)

        # Apply softmax to the logits to convert them to probabilities

        # x = x.view(-1)
        #Flatten the tensor to 1D if necessary #TODO: ask here if this makes sense

        return x