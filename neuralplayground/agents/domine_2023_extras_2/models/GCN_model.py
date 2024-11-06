import torch.nn as nn
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv,global_mean_pool
# TODO(clementine): set up object oriented GNN classes (eventually)
class GCNModel(nn.Module):
    def __init__(self, num_hidden, num_feature, num_layers, num_message_passing_steps, residual):
        super(GCNModel, self).__init__()
        self.num_message_passing_steps = num_message_passing_steps
        self.conv_1 = GCNConv(num_feature, num_hidden)
        self.conv_layers = nn.ModuleList([GCNConv(num_hidden, num_hidden) for _ in range(num_message_passing_steps)])
        # Output layer with size 2 for binary classification logits
        self.fc = nn.Linear(num_hidden, 2)
        self.residual = residual

    def forward(self, node, edges, edges_attr):
        x, edge_index = node, edges
        x = self.conv_1(x, edge_index, edges_attr)
        x = torch.relu(x)
        for i, conv in enumerate(self.conv_layers):
            x_res = x
            x = conv(x, edge_index, edges_attr)
            if self.residual:
                x += x_res
            x = torch.relu(x)
        # The output layer now produces 2 logits for each graph node
        x = self.fc(x)
        # Pooling to get a graph-level representation
        x = global_mean_pool(x, batch=None)
        # x = x.view(-1)
        #Flatten the tensor to 1D if necessary #TODO: ask here if this makes sense
        return x

# This is juste to test what happens when we don't have the message passing layers


class MLP(nn.Module):
        def __init__(self, num_hidden, num_feature, num_layers, num_message_passing_steps, residual):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(num_feature, num_hidden)  # First fully connected layer
            self.fc2 = nn.Linear(num_hidden, 2)
        def forward(self, node, edges, edges_attr):
            x, edge_index = node, edges
            # The output layer now produces 2 logits for each graph node
            x = torch.relu(self.fc1(x))  # Apply ReLU activation to first layer's output
            x = self.fc2(x)  # Pass through the second layer
            # Pooling to get a graph-level representation
            x = global_mean_pool(x, batch=None)
            return x

class GCNModel_2(nn.Module):
    def __init__(self, num_hidden, num_feature, num_layers, num_message_passing_steps, residual):
        super(GCNModel_2, self).__init__()
        self.num_message_passing_steps = num_message_passing_steps - 1
        self.conv_1 = GCNConv(num_feature, num_hidden)
        self.conv_layers = nn.ModuleList([GCNConv(num_hidden, num_hidden) for _ in range(num_message_passing_steps)])
        # Output layer with size 2 for binary classification logits
        self.fc = nn.Linear(num_hidden, 2)
        self.residual = residual

    def forward(self, node, edges,edges_attr):


        x, edge_index = node, edges
        x = self.conv_1(x, edge_index)
        x = torch.relu(x)
        for i, conv in enumerate(self.conv_layers):
            x_res = x
            x = conv(x, edge_index)
            if self.residual:
                x += x_res
            x = torch.relu(x)
        x = self.fc(x)
        x = global_mean_pool(x, batch=None)
        return x
