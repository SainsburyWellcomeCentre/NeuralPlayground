import matplotlib.pyplot as plt
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
import torch
from scipy.ndimage import convolve1d

def plot_2dgraphs(edges_list, node_features, edge_features_list, subplots_titles, path=None, colorscale='Plasma', size=5, show=True):
    """
    Plot 2D graphs with Plotly, generating node positions using a graph layout algorithm.

    :param edges_list: list of edge tensors (PyTorch format with shape [2, num_edges])
    :param node_features: list of node features or node colors for each graph
    :param edge_features_list: list of edge features for each graph (representing relative distances)
    :param subplots_titles: titles for the subplots
    :param colorscale: colorscale for the plot
    :param size: size of the points (nodes)
    :param show: whether to display the plot
    :param path: where to save the plot if specified
    :return: plotly figure
    """
    # Create the figure with subplots
    fig = make_subplots(
        rows=1, cols=len(edges_list),
        subplot_titles=subplots_titles,
        specs=[[{'type': 'xy'} for _ in range(len(edges_list))]]
    )

    fig.layout.coloraxis.colorscale = colorscale

    for i, edges in enumerate(edges_list):
        # Create a directed graph from the edges
        G = nx.DiGraph()  # Use DiGraph for directed edges

        # Convert PyTorch edges tensor and edge features to NumPy arrays
        edges_np = edges.cpu().numpy()  # Convert PyTorch tensor to NumPy array
        edge_features_np = edge_features_list[i].cpu().numpy()  # Convert edge features to NumPy if needed

        # Add edges to the directed graph G
        G.add_edges_from(edges_np.T)  # Add edges from the PyTorch format (transpose is needed)

        # Generate node positions using a layout (e.g., Fruchterman-Reingold force-directed layout)
        pos = nx.spring_layout(G, seed=42)  # You can use other layouts like kamada_kawai_layout

        # Convert positions into a numpy array for plotting
        pos_arr = np.array([pos[node] for node in G.nodes()])

        # Add the nodes as scatter plots
        fig.add_trace(
            go.Scatter(x=pos_arr[:, 0], y=pos_arr[:, 1],
                       mode='markers',
                       marker=dict(
                           size=size,
                           color=node_features[i],  # Use node features for color
                           coloraxis="coloraxis",
                           showscale=False)),
            row=1, col=i + 1)

        # Add directed edges with arrows and relative distances as edge features
        for j, (src, dst) in enumerate(G.edges()):
            x0, y0 = pos[src]
            x1, y1 = pos[dst]

            # Calculate arrowhead position
            arrow_fraction = 0.95  # Determines how far the arrowhead is from the start (95% of the way)
            arrow_x = x0 + arrow_fraction * (x1 - x0)
            arrow_y = y0 + arrow_fraction * (y1 - y0)

            # Use edge feature (relative distance) to modify line width
            edge_color = 'rgb(125,125,125)'  # Default edge color
            edge_width = max(1, edge_features_np[j])  # Use edge feature (relative distance) to determine width

            # Add the directed edge as a line
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1],
                           mode='lines',
                           line=dict(color=edge_color, dash='solid', width=edge_width),
                           showlegend=False),
                row=1, col=i + 1)

            # Add the arrow for the directed edge
            fig.add_trace(
                go.Scatter(x=[arrow_x], y=[arrow_y],
                           mode='markers',
                           marker=dict(size=5, color=edge_color, symbol='triangle-up'),
                           showlegend=False),
                row=1, col=i + 1)

    # Add colorbar if needed
    colorbar_trace = go.Scatter(x=[None], y=[None], mode='markers',
                                marker=dict(showscale=True, coloraxis="coloraxis",
                                            colorbar=dict(thickness=10, outlinewidth=0)),
                                hoverinfo='none')

    fig.add_trace(colorbar_trace, row=1, col=len(edges_list))  # Add colorbar to the last subplot

    # Show the figure if requested
    if show:
        fig.show()

    # Save the figure if a path is provided
    if path:
        fig.write_image(path)

    return fig


def color_map(output):
    vmin = min(output)
    vmax = max(output)
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.jet, norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm._A = []
    return sm

def plot_curves(curves, path, title, legend_labels=None, x_label=None, y_label=None, time_steps=100):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title)

    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)

    colormap = plt.get_cmap("viridis")

    # Use numpy linspace to create the values array
    values = np.linspace(0, 1, len(curves))

    # Map the values to colors using the chosen colormap
    colors = [colormap(value) for value in values]

    # Create a simple kernel for convolution (moving average)
    kernel = np.ones(time_steps) / time_steps

    for i, curve in enumerate(curves):
        label = legend_labels[i] if legend_labels else None
        color = colors[i % len(colors)]

        # Convert PyTorch tensors to NumPy arrays for plotting
        if isinstance(curve, torch.Tensor):
            curve = curve.detach().numpy()

        # Apply convolution to smooth the curve over the specified time steps
        curve = convolve1d(curve, kernel, mode='reflect')

        ax.plot(curve, label=label, color=color)

    if legend_labels:
        ax.legend()
    plt.savefig(path)
    plt.show()
    plt.close()
def plot_curves_2(curves, std_devs=None, path=None, title=None, legend_labels=None, x_label=None, y_label=None, time_steps=1):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title)

    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)

    colormap = plt.get_cmap("viridis")

    # Use numpy linspace to create the values array
    values = np.linspace(0, 1, len(curves))

    # Map the values to colors using the chosen colormap
    colors = [colormap(value) for value in values]

    # Create a simple kernel for convolution (moving average)
    kernel = np.ones(time_steps) / time_steps

    for i, curve in enumerate(curves):
        label = legend_labels[i] if legend_labels else None
        color = colors[i % len(colors)]

        # Convert PyTorch tensors to NumPy arrays for plotting
        if isinstance(curve, torch.Tensor):
            curve = curve.detach().numpy()

        # Apply convolution to smooth the curve over the specified time steps
        curve = convolve1d(curve, kernel, mode='reflect')

        # Plot the smoothed average curve
        ax.plot(curve, label=label, color=color)

        # If std_devs are provided, plot the shaded region for the standard deviation
        if std_devs is not None:
            std_dev = std_devs[i]

            # Convert std_dev to numpy if it is a PyTorch tensor
            if isinstance(std_dev, torch.Tensor):
                std_dev = std_dev.detach().numpy()

            # Apply convolution to smooth the standard deviation over the specified time steps
            std_dev = convolve1d(std_dev, kernel, mode='reflect')

            # Shaded region (curve ± std deviation)
            ax.fill_between(np.arange(len(curve)), curve - std_dev, curve + std_dev, color=color, alpha=0.3)

    if legend_labels:
        ax.legend()

    # Save plot to file if path is provided
    if path:
        plt.savefig(path)

    plt.show()
    plt.close()