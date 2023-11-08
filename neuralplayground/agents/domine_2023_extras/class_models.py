import haiku as hk
import jraph

# TODO(clementine): set up object oriented GNN classes (eventually)


def get_forward_function(num_hidden, num_layers, num_message_passing_steps,add_residual=True, use_layer_norm =False):
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
            embed_node_fn=hk.Linear(output_size=num_hidden),
        )(x)

        # Apply rounds of message passing.
        message_passing = []
        layer_output =[]
        for n in range(num_message_passing_steps):
            if add_residual:
                previous_x = x  # Store the current state for the residual connection
                x = message_passing_layer(x, edge_mlp_sizes, node_mlp_sizes, use_layer_norm)
                x = x._replace(nodes=x.nodes + previous_x.nodes, edges=x.edges + previous_x.edges)
            #layer_output +=  message_passing_layer(x, edge_mlp_sizes, node_mlp_sizes, use_layer_norm)
            else:
                x = message_passing_layer(x, edge_mlp_sizes, node_mlp_sizes, use_layer_norm)
            message_passing.append(x)
            #x = message_passing_layer(x, edge_mlp_sizes, node_mlp_sizes)
            #message_passing.   append(x)
        # Map features to desired feature size.
        x = jraph.GraphMapFeatures(
            embed_edge_fn=hk.Linear(output_size=edge_output_size),
            embed_node_fn=hk.Linear(output_size=node_output_size),
        )(x)
        return x, message_passing

    return _forward

def mlp(edge_mlp_sizes, use_layer_norm):
  sequential_modules = [hk.nets.MLP(output_sizes=edge_mlp_sizes)]
  if use_layer_norm:
    #TODO: Clementine Domine check if this is the right axis
    sequential_modules.append(hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True,create_offset=True))
  return hk.Sequential(sequential_modules)

def message_passing_layer(x, edge_mlp_sizes, node_mlp_sizes, use_layer_norm):
    update_edge_fn = jraph.concatenated_args(mlp(edge_mlp_sizes, use_layer_norm))
    update_node_fn = jraph.concatenated_args(mlp(node_mlp_sizes, use_layer_norm))
    x = jraph.GraphNetwork(
        update_edge_fn=update_edge_fn, update_node_fn=update_node_fn
    )(x)
    return x

#def message_passing_layer(x, edge_mlp_sizes, node_mlp_sizes):
#   update_edge_fn = jraph.concatenated_args(hk.nets.MLP(output_sizes=edge_mlp_sizes))
#   update_node_fn = jraph.concatenated_args(hk.nets.MLP(output_sizes=node_mlp_sizes))
#  x = jraph.GraphNetwork(
#     update_edge_fn=update_edge_fn, update_node_fn=update_node_fn
#)(x)
#return x
