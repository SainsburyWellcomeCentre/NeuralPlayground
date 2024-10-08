
import numpy as np
import torch
import scipy.spatial as spatial

from neuralplayground.agents.domine_2023_extras.class_utils import rng_sequence_from_rng

def sample_graph(train=True):
    Node_Features = torch.tensor([
        [0.1, 0.5, 0.3],
        [0.4, 0.7, 0.9],
        [0.2, 0.6, 0.8],
        [0.3, 0.9, 0.7]
    ])
    Adj= torch.tensor([[0, 1, 2], [1, 2, 3]])
    return Node_Features, Adj

def sample_graph_c(
    rng,
    batch_size,
    feature_position,
    weighted,
    nx_min,
    nx_max,
    grid=False,
    dist_cutoff = 10,
    n_std_dist_cutoff = 5,
    ny_min=None,
    ny_max=None,
):

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

    dist_cutoff = dist_cutoff
    n_std_dist_cutoff =  n_std_dist_cutoff
    rng_seq = rng_sequence_from_rng(rng)
    ny_min = ny_min or nx_min
    ny_max = ny_max or nx_max
    #TODO Make sure this is ok padding
    max_n = ny_max * nx_max * batch_size
    max_e = max_n * max_n
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
        if grid:
            nx_graph = get_grid_adjacency(n_x, n_y)
            # TODO: Check that this is working fine
            n=0
            for node in nx_graph.nodes:
                nx_graph.nodes[node]['pos'] = np.array(node, dtype=float)
            i_start_1 = jax.random.randint(next(rng_seq), shape=(1,), minval=0, maxval=n_x)
            i_start_2 = jax.random.randint(next(rng_seq), shape=(1,), minval=0, maxval=n_y)
            i_end_1 = jax.random.randint(next(rng_seq), shape=(1,), minval=0, maxval=n_x)
            i_end_2 = jax.random.randint(next(rng_seq), shape=(1,), minval=0, maxval=n_y)
            start = tuple([i_start_1.tolist()[0], i_start_2.tolist()[0]])
            end = tuple([i_end_1.tolist()[0], i_end_2.tolist()[0]])
            node_number_start = (i_start_1) * n_y + (i_start_2)
            node_number_end = (i_end_1) * n_y + (i_end_2)
        else:
            #TODO:Positon that have position information ; oriiginal  use the pos here bellow check that the new update of the positon is working fine
            seed = jax.random.randint(next(rng_seq), shape=(1,), minval=0, maxval=1000)
            nx_graph, pos = random_geometric_delaunay_graph_generator(int(max_n/batch_size)-1, seed[0], dist_cutoff, n_std_dist_cutoff)
            for node, pos_i in zip(nx_graph.nodes, pos):
                nx_graph.nodes[node]['pos'] = pos_i
            i_start_1 = jax.random.randint(next(rng_seq), shape=(1,), minval=0, maxval=int(max_n/batch_size)-1).tolist()
            i_end_1 = jax.random.randint(next(rng_seq), shape=(1,), minval=0, maxval=int(max_n/batch_size)-1).tolist()
            start = i_start_1[0]
            end = i_end_1[0]
            node_number_start = i_start_1
            node_number_end = i_end_1

        if weighted:
            weights = add_weighted_edge(int(nx_graph.number_of_edges()), 1, rng_seq)
            r = 0
            for i, j in nx_graph.edges:
                r = r + 1
                nx_graph[i][j]["weight"] = weights[r]
            nodes_on_shortest_path_indexes = nx.shortest_path(
                nx_graph, start, end, weight="weight"
            )
        else:
            nodes_on_shortest_path_indexes = nx.shortest_path(
                nx_graph, start, end
            )

        # make it a node feature of the input graph if a node is a start/end node
        input_node_features = jnp.zeros((int(nx_graph.number_of_nodes()), 1))

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
            input_node_features = jnp.concatenate( (input_node_features, node_positions), axis=1 )

        nx_graph = nx.DiGraph(nx_graph)
        distance = (node_positions[receivers] - node_positions[senders])
        if weighted:
            edges_features = jnp.array(
                [nx_graph[s][r]["weight"] for s, r in nx_graph.edges]
            )
            weighted_distance =edges_features *distance
            graph = jraph.GraphsTuple(
                nodes=input_node_features,
                senders=senders,
                receivers=receivers,
                edges=weighted_distance,
                n_node=jnp.array([n_node], dtype=int),
                n_edge=jnp.array([n_edge], dtype=int),
                globals=global_context,
            )
        else:
            #if grid:
                # edge_displacement=np.sum(abs(edge_displacements),1).reshape(-1, 1)
                #distance = jnp.sqrt(jnp.sum((edge_displacements) ** 2, 1)).reshape(-1, 1)
            graph = jraph.GraphsTuple(
                nodes=input_node_features,
                senders=senders,
                edges=distance,
                receivers=receivers,
                n_node=jnp.array([n_node], dtype=int),
                n_edge=jnp.array([n_edge], dtype=int),
                globals=global_context,
            )
            #else:
                #edges_features = jnp.array(
                 #   [nx_graph[s][r]["weight"] for s, r in nx_graph.edges]
                #).reshape(-1,1)
                #TODO: make sure that this correction is actualy correct

                #graph = jraph.GraphsTuple(
                    #nodes=input_node_features,
                    #senders=senders,
                # receivers=receivers,
                # edges=distance,
                # n_node=jnp.array([n_node], dtype=int),
                # n_edge=jnp.array([n_edge], dtype=int),
                #  globals=global_context,
                # )

        graphs.append(graph)
        nodes_on_shortest_labels = jnp.zeros((n_node, 1))

        if grid:
            for i in nodes_on_shortest_path_indexes:
                    l = jnp.argwhere(
                        jnp.all((node_positions - jnp.asarray(i)) == 0, axis=1)
                    )
                    nodes_on_shortest_labels = nodes_on_shortest_labels.at[l[0, 0]].set(1)
            target.append(nodes_on_shortest_labels)  # set start node feature
        else:
            for i in nodes_on_shortest_path_indexes:

                #TODO: Check that this is the right indexing
                nodes_on_shortest_labels = nodes_on_shortest_labels.at[i].set(1)
            target.append(nodes_on_shortest_labels)  # set start node feature

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


"""Functions for Random Delaunay Triangulations."""

