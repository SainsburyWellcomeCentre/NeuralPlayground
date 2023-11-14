# deepmind (i think) package for optimization
import jax
import jax.numpy as jnp
import jraph
import networkx as nx
from neuralplayground.agents.domine_2023_extras.class_utils import rng_sequence_from_rng


def get_grid_adjacency(n_x, n_y, atol=1e-1):
    return nx.grid_2d_graph(n_x, n_y)  # Get directed grid graph


def sample_padded_grid_batch_shortest_path(
    rng,
    batch_size,
    feature_position,
    weighted,
    nx_min,
    nx_max,
    ny_min=None,
    ny_max=None,
):
    rng_seq = rng_sequence_from_rng(rng)
    """Sample a batch of grid graphs with variable sizes.
  Args:
    rng: jax.random.PRNGKey(integer_seed) object, random number generator
    batch_size: int, number of graphs to sample
    nx_min: minum size along x axis
    nx_max: maximum size along x axis
    ny_min: minum size along y axis (default: nx_min)
    ny_max: maximum size along y axis (default: ny_max)
  Returns:
    padded graph batch that can be fed into a jraph GNN.
  """
    ny_min = ny_min or nx_min
    ny_max = ny_max or nx_max
    max_n = ny_max * nx_max * batch_size
    max_e = max_n * 4
    # Sample grid dimensions
    x_rng = next(rng_seq)
    y_rng = next(
        rng_seq
    )  # need to "split" to advance the random number generator -- otherwise rng will be the same for all things sampled from it
    n_xs = jax.random.randint(x_rng, shape=(batch_size,), minval=nx_min, maxval=nx_max)
    n_ys = jax.random.randint(y_rng, shape=(batch_size,), minval=ny_min, maxval=ny_max)
    # Construct graphs with sampled dimensions.
    graphs = []
    target = []
    for n_x, n_y in zip(n_xs, n_ys):
        nx_graph = get_grid_adjacency(n_x, n_y)

        weights = add_weighted_edge(int(nx_graph.number_of_edges()), 1, rng_seq)
        r = 0
        for i, j in nx_graph.edges:
            r = r + 1
            nx_graph[i][j]["weight"] = weights[r]

        i_start_1 = jax.random.randint(next(rng_seq), shape=(1,), minval=0, maxval=n_x)
        i_start_2 = jax.random.randint(next(rng_seq), shape=(1,), minval=0, maxval=n_y)
        i_end_1 = jax.random.randint(next(rng_seq), shape=(1,), minval=0, maxval=n_x)
        i_end_2 = jax.random.randint(next(rng_seq), shape=(1,), minval=0, maxval=n_y)

        start = tuple([i_start_1.tolist()[0], i_start_2.tolist()[0]])
        end = tuple([i_end_1.tolist()[0], i_end_2.tolist()[0]])

        nodes_on_shortest_path_indexes_not_weighted = nx.shortest_path(
            nx_graph, start, end
        )
        nodes_on_shortest_path_indexes = nx.shortest_path(
            nx_graph, start, end, weight="weight"
        )

        # make it a node feature of the input graph if a node is a start/end node
        input_node_features = jnp.zeros((int(nx_graph.number_of_nodes()), 1))

        node_number_start = (i_start_1) * n_y + (i_start_2)
        node_number_end = (i_end_1) * n_y + (i_end_2)

        input_node_features = input_node_features.at[node_number_start, 0].set(
            1
        )  # set start node feature
        input_node_features = input_node_features.at[node_number_end, 0].set(
            1
        )  # set end node feature

        (
            senders,
            receivers,
            node_positions,
            edge_displacements,
            n_node,
            n_edge,
            global_context,
        ) = grid_networkx_to_graphstuple(nx_graph)

        if feature_position:
            input_node_features = jnp.concatenate(
                (input_node_features, node_positions), axis=1
            )

        nx_graph = nx.DiGraph(nx_graph)
        if weighted:
            edges_features = jnp.array(
                [nx_graph[s][r]["weight"] for s, r in nx_graph.edges]
            )
            graph = jraph.GraphsTuple(
                nodes=input_node_features,
                senders=senders,
                receivers=receivers,
                edges=edges_features,
                n_node=jnp.array([n_node], dtype=int),
                n_edge=jnp.array([n_edge], dtype=int),
                globals=global_context,
            )

        else:
            # TODO:Clementine: Chamge this line
            # edge_displacement=np.sum(abs(edge_displacements),1).reshape(-1, 1)
            distance = jnp.sqrt(jnp.sum((edge_displacements) ** 2, 1)).reshape(-1, 1)
            graph = jraph.GraphsTuple(
                nodes=input_node_features,
                senders=senders,
                edges=distance,
                receivers=receivers,
                n_node=jnp.array([n_node], dtype=int),
                n_edge=jnp.array([n_edge], dtype=int),
                globals=global_context,
            )

        graphs.append(graph)
        nodes_on_shortest_labels = jnp.zeros((n_node, 1))
        if weighted:
            for i in nodes_on_shortest_path_indexes:
                l = jnp.argwhere(
                    jnp.all((node_positions - jnp.asarray(i)) == 0, axis=1)
                )
                nodes_on_shortest_labels = nodes_on_shortest_labels.at[l[0, 0]].set(1)
            target.append(nodes_on_shortest_labels)  # set start node feature
        else:
            for i in nodes_on_shortest_path_indexes_not_weighted:
                l = jnp.argwhere(
                    jnp.all((node_positions - jnp.asarray(i)) == 0, axis=1)
                )
                nodes_on_shortest_labels = nodes_on_shortest_labels.at[l[0, 0]].set(1)
            target.append(nodes_on_shortest_labels)

    targets = jnp.concatenate(target)
    target_pad = jnp.zeros(((max_n - len(targets)), 1))
    padded_target = jnp.concatenate((targets, target_pad), axis=0)
    graph_batch = jraph.batch(graphs)
    padded_graph_batch = jraph.pad_with_graphs(
        graph_batch, n_node=max_n, n_edge=max_e, n_graph=len(graphs) + 1
    )

    return (
        padded_graph_batch,
        jnp.asarray(padded_target),
    )


def grid_networkx_to_graphstuple(nx_graph):
    """Get edges for a grid graph."""
    nx_graph = nx.DiGraph(nx_graph)
    node_positions = jnp.array(nx_graph.nodes)
    node_to_inds = {n: i for i, n in enumerate(nx_graph.nodes)}
    senders_receivers = [(node_to_inds[s], node_to_inds[r]) for s, r in nx_graph.edges]
    edge_displacements = jnp.array(
        [jnp.array(r) - jnp.array(s) for s, r in nx_graph.edges]
    )
    senders, receivers = zip(*senders_receivers)
    n_node = node_positions.shape[0]
    n_edge = edge_displacements.shape[0]
    return (
        jnp.array(senders, dtype=int),
        jnp.array(receivers, dtype=int),
        jnp.array(node_positions, dtype=float),
        jnp.array(edge_displacements, dtype=float),
        n_node,
        n_edge,
        jnp.zeros((1, 0), dtype=float),
    )


"Need to figure out how it is working with the batch thing, how it is working" "Figure out as well other types of weighting on the edges "
"Here I have a problem again because for one of them i have to have positive weights " "I think I will add it as a feature of the edges "


def add_weighted_edge(n_edge, sigma_on_edge_weight_noise, rng_seq):
    weights = jnp.zeros((n_edge, 1))
    for k in range(n_edge):
        rng = next(rng_seq)
        r = jax.random.uniform(rng, shape=(1,))
        weight = jnp.asarray(jax.lax.round(10 * r[0]) * 0.1 + int(1))
        weights = weights.at[k, 0].set(weight)
        # edge_displacement = edge_displacement.at[k,l].set(edge_displacement[k][l] + weight)    # weights=sigma_on_edge_weight_noise * np.random.rand() Because nedd postiove and add as features and need ot be used by the neural networks :)
    return weights
