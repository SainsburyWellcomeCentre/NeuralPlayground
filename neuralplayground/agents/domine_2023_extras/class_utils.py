# @title Make rng sequence generator

import jax
import jraph
import networkx as nx
import torch


def rng_sequence(seed):
    # Makes a generator that produces novel RNGs
    rng = jax.random.PRNGKey(seed)
    while True:
        rng, _ = jax.random.split(rng)
        yield rng


def rng_sequence_from_rng(rng):
    # Makes a generator that produces novel RNGs sequence from a given RNG
    # use rng_seq = rng_sequence_from_rng(rng); rng_new = next(rng_seq)
    while True:
        rng, _ = jax.random.split(rng)
        yield rng


def convert_jraph_to_networkx_graph(
    jraph_graph: jraph.GraphsTuple, number_graph_batch
) -> nx.Graph:
    nodes, edges, receivers, senders, _, _, _ = jraph_graph
    node_padd = 0
    edges_padd = 0
    for i in range(number_graph_batch):
        node_padd = node_padd + jraph_graph.n_node[i]
        edges_padd = edges_padd + jraph_graph.n_edge[i]
    nx_graph = nx.DiGraph()
    if nodes is None:
        for n in range(jraph_graph.n_node[number_graph_batch]):
            nx_graph.add_node(n)
    else:
        for n in range(jraph_graph.n_node[number_graph_batch]):
            nx_graph.add_node(n, node_feature=nodes[node_padd + n])
        for e in range(jraph_graph.n_edge[number_graph_batch]):
            nx_graph.add_edge(
                int(senders[edges_padd + e]) - node_padd,
                int(receivers[edges_padd + e] - node_padd),
                edge_feature=edges[edges_padd + e],
            )
    return nx_graph


def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print(
            "WARNING: For this notebook to perform best, "
            "if possible, in the menu under `Runtime` -> "
            "`Change runtime type.`  select `GPU` "
        )
    else:
        print("GPU is enabled in this notebook.")


def get_activations_graph_n(
    node_colour,
    graph,
    number_graph_batch=0,
):
    node_padd = get_node_pad(graph, number_graph_batch)
    output = node_colour[node_padd : node_padd + graph.n_node[number_graph_batch]]
    return output


# maybe actually change the node pad to a node padd function


def get_node_pad(graph, i):
    node_padd = 0
    for j in range(i):
        node_padd = node_padd + graph.n_node[j]
    return node_padd


def update_outputs_test(outputs, indices):
    outputs_wse = outputs[0].nodes
    for ind in indices:
        outputs_wse = outputs_wse.at[ind].set(0)
    return outputs_wse
