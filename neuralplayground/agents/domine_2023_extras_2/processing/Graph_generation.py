
import numpy as np
import torch
import torchvision
from torchvision import transforms
from neuralplayground.agents.domine_2023_extras.class_utils import rng_sequence_from_rng

def create_random_matrix(rows, cols,seed, low=0, high=1):
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

def get_omniglot_items(n):
    # Define transformations for the images
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Resize((28, 28)),  # Resize to 28x28 (if needed)
        transforms.RandomRotation(20),  # Randomly rotate the image by up to 20 degrees
        transforms.RandomHorizontalFlip(p=0.5)  # Randomly flip the image horizontally with 50% probability
    ])

    # Load the Omniglot dataset with the defined transformation
    dataset = torchvision.datasets.Omniglot(
        root='./data',
        background=True,  # Use background set, set to False to use evaluation set
        download=True,
        transform=transform  # Apply the defined transformations
    )
    # Initialize list to hold sampled features
    features = []

    # Randomly sample n items from the dataset
    for _ in range(n):
        # Get a random index in the range of the dataset length
        random_idx = np.random.randint(0, len(dataset))
        # Retrieve the item (image, label)
        image, label = dataset[random_idx]
        # Convert the image to a numpy array (already transformed)
        image_vector = image.numpy()
        # Reshape the image from (1, 28, 28) to (1, n), where n is 28 * 28 = 784
        image_vector = image_vector.flatten()
        # Append the image to the feature list (image is already transformed and a tensor)
        features.append(image_vector)  # Convert to numpy array for convenience
    # Convert list of features to numpy array
    features_array = np.array(features)
    return features_array

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
        return torch.tensor([1], dtype=torch.long)
    elif source < sink:
        return  torch.tensor([0], dtype= torch.long)
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
def sample_random_graph(num_features, num_nodes,seed):
    # This is a graph with edges feature and random features
    node_features = torch.tensor(create_random_matrix(num_nodes,num_features,seed))
    edges , edge_features_tensor =  create_line_graph_edge_list_with_features(num_nodes)
    input_node_features = np.zeros((int(num_nodes), 2))
    sink, source = generate_source_and_sink(num_nodes)
    input_node_features[source, 0] = 1  # Set source node feature
    input_node_features[sink, 1] = 1  # Set sink node feature
    # Concatenate the feature matrices along the feature dimension (axis=1)
    combined_node_features = np.concatenate([node_features, input_node_features], axis=1)
    # Convert combined node features back to a tensor
    node_features = torch.tensor(combined_node_features, dtype=torch.float32)
    return node_features, edges, edge_features_tensor, source, sink

#TODO: we need to merge this two potentially

def sample_omniglot_graph(num_nodes,seed):
    # This is a graph with edges feature and omniglot features
    node_features = torch.tensor(get_omniglot_items(num_nodes))
    edges , edge_features_tensor = create_line_graph_edge_list_with_features(num_nodes)
    input_node_features = np.zeros((int(num_nodes), 2))
    sink, source = generate_source_and_sink(num_nodes)
    input_node_features[source, 0] = 1  # Set source node feature
    input_node_features[sink, 1] = 1  # Set sink node feature
    # Concatenate the feature matrices along the feature dimension (axis=1)
    combined_node_features = np.concatenate([node_features, input_node_features], axis=1)
    # Convert combined node features back to a tensor
    node_features = torch.tensor(combined_node_features, dtype=torch.float32)
    return node_features, edges, edge_features_tensor, source, sink

def sample_random_graph_position(num_features, num_nodes,seed):
    # This is a graph with edges feature and position features
    node_features = torch.tensor(create_random_matrix(num_nodes,num_features))
    edges , edge_features_tensor =  create_line_graph_edge_list_with_features(num_nodes)
    input_node_features = np.zeros((int(num_nodes), 2))
    sink, source = generate_source_and_sink(num_nodes)
    input_node_features[source, 0] = 1  # Set source node feature
    input_node_features[sink, 1] = 1  # Set sink node feature
    # Concatenate the feature matrices along the feature dimension (axis=1)
    combined_node_features = np.concatenate([node_features, input_node_features], axis=1)
    position =  torch.tensor([np.arange(0, num_nodes)])
    combined_node_features_pos = np.concatenate([combined_node_features, position.T], axis=1)
    # Convert combined node features back to a tensor
    node_features = torch.tensor(combined_node_features_pos , dtype=torch.float32)
    return node_features, edges, edge_features_tensor, source, sink

def sample_random_graph_position_no_edges(num_features, num_nodes,seed):
    # This is a graph with no edges feature but position features
    node_features = torch.tensor(create_random_matrix(num_nodes,num_features))
    edges , edge_features_tensor =  create_line_graph_edge_list_with_features(num_nodes)
    input_node_features = np.zeros((int(num_nodes), 2))
    sink, source = generate_source_and_sink(num_nodes)
    input_node_features[source, 0] = 1  # Set source node feature
    input_node_features[sink, 1] = 1  # Set sink node feature
    # Concatenate the feature matrices along the feature dimension (axis=1)
    combined_node_features = np.concatenate([node_features, input_node_features], axis=1)
    # Convert combined node features back to a tensor
    node_features = torch.tensor(combined_node_features, dtype=torch.float32)
    return node_features, edges, source, sink

#TODO: we need to merge this into one function because this is ungly
