
import numpy as np
import torch
import scipy.spatial as spatial

from neuralplayground.agents.domine_2023_extras.class_utils import rng_sequence_from_rng

def create_random_matrix(rows, cols, low=0, high=1):
    """
    Generates a random matrix with the specified dimensions.
    Parameters:
    rows (int): Number of rows in the matrix.
    cols (int): Number of columns in the matrix.
    low (float): The lower bound of the random values (inclusive). Default is 0.
    high (float): The upper bound of the random values (exclusive). Default is 1.
    Returns:
    numpy.ndarray: A matrix of shape (rows, cols) with random values.
    """
    return np.random.uniform(low, high, (rows, cols))


def create_line_graph_edge_list_with_features(num_nodes):
    """
    Creates a directed graph that represents a line where nodes can be traversed back and forth.
    Returns the edge list with shape (2, num_edges) and an edge feature vector where forward edges
    have a feature of 1 and backward edges have a feature of -1.

    :param num_nodes: The number of nodes in the graph
    :return: edge list as a torch tensor of shape (2, num_edges), edge features as torch tensor
    """
    edges = []
    edge_features = []

    # Create edges for a directed line graph
    for i in range(num_nodes - 1):
        # Add edge from node i to i+1 (forward) with feature 1
        edges.append([i, i + 1])
        edge_features.append(1)

        # Add edge from node i+1 to i (backward) with feature -1
        edges.append([i + 1, i])
        edge_features.append(-1)

    # Convert the list of edges to a numpy array and then to a torch tensor
    edges = np.array(edges).T  # Shape (2, num_edges)
    edge_tensor = torch.tensor(edges, dtype=torch.long)

    # Convert edge features to a torch tensor
    edge_features_tensor = torch.tensor(edge_features, dtype=torch.float32)

    return edge_tensor, edge_features_tensor

def sample_target(source, sink):
    """
    Compares the source and sink nodes and returns:
    - 1 if the source is greater than the sink
    - -1 if the source is smaller than the sink
    - 0 if the source is equal to the sink
    :param source: The source node index
    :param sink: The sink node index
    :return: 1, -1, or 0 based on the comparison
    """
    if source > sink:
        return torch.tensor([1], dtype=torch.float32)
    elif source < sink:
        return  torch.tensor([0], dtype=torch.float32)
    else:
        return 'error'


def generate_source_and_sink(num_nodes):
    """
    Generates a source and a sink such that they are not the same.
    :param num_nodes: The total number of nodes
    :return: A tuple (source, sink) where source != sink
    """
    source = np.random.randint(0, num_nodes)
    # Ensure sink is not equal to source
    sink = np.random.randint(0, num_nodes)
    while sink == source:
        sink = np.random.randint(0, num_nodes)

    return source, sink
def sample_graph(num_features, num_nodes):
    node_features = torch.tensor(create_random_matrix(num_features, num_nodes))
    edges , edge_features_tensor =  create_line_graph_edge_list_with_features(num_nodes)
    input_node_features = np.zeros((int(num_nodes), 2))
    sink, source = generate_source_and_sink(num_nodes)
    input_node_features[source, 0] = 1  # Set source node feature
    input_node_features[sink, 1] = 1  # Set sink node feature
    # Concatenate the feature matrices along the feature dimension (axis=1)
    combined_node_features = np.concatenate([node_features.T, input_node_features], axis=1)
    # Convert combined node features back to a tensor
    node_features = torch.tensor(combined_node_features, dtype=torch.float32)
    return node_features, edges, edge_features_tensor, source, sink

