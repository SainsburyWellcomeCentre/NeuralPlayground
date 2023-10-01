import haiku as hk
import jraph
import jax.numpy as jnp
from plotting_utils import plot_message_passing_layers

#TODO(clementine): set up object oriented GNN classes (eventually)

def get_forward_function(num_hidden, num_layers,num_message_passing_steps):
    """Get function that performs a forward call on a simple GNN."""
    def _forward(x):
        """Forward pass of a simple GNN."""
        node_output_size = 1
        edge_output_size = 1

        # Set up MLP parameters for node/edge updates
        node_mlp_sizes = [num_hidden] * num_layers
        edge_mlp_sizes = [num_hidden] * num_layers

        # Map features to desired feature size.
        x = jraph.GraphMapFeatures(
            embed_edge_fn=hk.Linear(output_size=num_hidden),
            embed_node_fn=hk.Linear(output_size=num_hidden))(x)

        # Apply rounds of message passing.
        message_passing=[]
        for n in range(num_message_passing_steps):
            x = message_passing_layer(x, edge_mlp_sizes, node_mlp_sizes)
            message_passing.append(x)

        # Map features to desired feature size.
        x = jraph.GraphMapFeatures(
            embed_edge_fn=hk.Linear(output_size=edge_output_size),
            embed_node_fn=hk.Linear(output_size=node_output_size))(x)

        return x , message_passing
    return _forward

def message_passing_layer(x, edge_mlp_sizes, node_mlp_sizes):
    update_edge_fn = jraph.concatenated_args(hk.nets.MLP(output_sizes=edge_mlp_sizes))
    update_node_fn = jraph.concatenated_args(hk.nets.MLP(output_sizes=node_mlp_sizes))
    x = jraph.GraphNetwork(update_edge_fn=update_edge_fn, update_node_fn=update_node_fn)(x)
    return x

