# @title Make rng sequence generator
import matplotlib.pyplot as plt

import networkx as nx
from neuralplayground.agents.domine_2023_extras.class_utils import (
    convert_jraph_to_networkx_graph,
    get_activations_graph_n,
    get_node_pad,
)


def plot_input_target_output(
    inputs, targets, outputs, graph, n, edege_lables, save_path, title
):
    # minim 2 otherwise it breaks
    rows = ["{}".format(row) for row in ["Input", "Target", "Outputs"]]
    fig, axes = plt.subplots(3, n)
    fig.set_size_inches(15, 15)
    for i in range(n):
        nx_graph = convert_jraph_to_networkx_graph(graph, i)
        pos = nx.spring_layout(nx_graph, iterations=100, seed=39775)
        input = get_activations_graph_n(inputs, graph, i)
        target = get_activations_graph_n(targets, graph, i)
        output = get_activations_graph_n(outputs, graph, i)
        axes[0, i].title.set_text("Input")
        axes[1, i].title.set_text("Target")
        axes[2, i].title.set_text("Output")
        u = 0
        # but I also need to make sure depending on what i , the papded thing for plittin and
        if edege_lables:
            node_padd = get_node_pad(graph, i)
            u = 0
            labels = nx.get_edge_attributes(nx_graph, "edge_feature")
            nx.draw_networkx_edge_labels(
                nx_graph, pos=pos, edge_labels=labels, ax=axes[0, i]
            )
            nx.draw_networkx_edge_labels(
                nx_graph, pos=pos, edge_labels=labels, ax=axes[1, i]
            )
            nx.draw_networkx_edge_labels(
                nx_graph, pos=pos, edge_labels=labels, ax=axes[2, i]
            )

        nx.draw(
            nx_graph,
            pos=pos,
            with_labels=True,
            node_size=200,
            node_color=input,
            font_color="white",
            cmap=plt.cm.jet,
            ax=axes[0, i],
        )
        plt.colorbar(color_map(input), ax = axes[0, i])

        nx.draw(
            nx_graph,
            pos=pos,
            with_labels=True,
            node_size=200,
            node_color=target,
            font_color="white",
            cmap=plt.cm.jet,
            ax=axes[1, i],
        )
        plt.colorbar(color_map(target),ax = axes[1, i])


        nx.draw(
            nx_graph,
            pos=pos,
            with_labels=True,
            node_size=200,
            node_color=output,
            font_color="white",
            cmap=plt.cm.jet,
            ax=axes[2, i],
        )
        plt.colorbar(color_map (output),ax = axes[2, i])

    plt.suptitle(title)
    plt.savefig(save_path)
    plt.close()

def color_map (output):
    vmin = min(output)
    vmax = max(output)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    return sm

def plot_message_passing_layers(
    inputs,
    activations,
    targets,
    outputs,
    graph,
    n,
    n_message_passing,
    edege_lables,
    save_path,
    title,
):
    # minim 2 otherwise it breaks
    fig, axes = plt.subplots(n_message_passing + 3, n)
    fig.set_size_inches(15, 15)
    for j in range(n):
        nx_graph = convert_jraph_to_networkx_graph(graph, j)
        pos = nx.spring_layout(nx_graph, iterations=100, seed=39775)
        if edege_lables:
            labels = nx.get_edge_attributes(nx_graph, "edge_feature")
        for i in range(n_message_passing + 3):
            if edege_lables:
                nx.draw_networkx_edge_labels(
                    nx_graph, pos=pos, edge_labels=labels, ax=axes[i, j]
                )
            if i == (n_message_passing + 2):
                axes[i, j].title.set_text("input")
                input = get_activations_graph_n(inputs, graph, j)
                nx.draw(
                    nx_graph,
                    pos=pos,
                    with_labels=True,
                    node_size=200,
                    node_color=input,
                    font_color="white",
                    cmap=plt.cm.jet,
                    ax=axes[i, j],
                )
                plt.colorbar(color_map(input), ax=axes[i, j])
            elif i == (n_message_passing + 1):
                axes[i, j].title.set_text("target")
                target = get_activations_graph_n(targets, graph, j)
                nx.draw(
                    nx_graph,
                    pos=pos,
                    with_labels=True,
                    node_size=200,
                    node_color=target,
                    font_color="white",
                    cmap=plt.cm.jet,
                    ax=axes[i, j],
                )
                plt.colorbar(color_map(input), ax=axes[i, j])
            elif i == (n_message_passing):
                axes[i, j].title.set_text("output")
                output = get_activations_graph_n(outputs, graph, j)
                nx.draw(
                    nx_graph,
                    pos=pos,
                    with_labels=True,
                    node_size=200,
                    node_color=output,
                    font_color="white",
                    cmap=plt.cm.jet,
                    ax=axes[i, j],
                )
                plt.colorbar(color_map(input), ax=axes[i, j])
            else:
                activation = activations[i]
                axes[i, j].title.set_text(
                    "graph_" + str(j) + "message_passing_" + str(i)
                )
                input = get_activations_graph_n(
                    activation.nodes[:, j].tolist(), graph, j
                )
                nx.draw(
                    nx_graph,
                    pos=pos,
                    with_labels=True,
                    node_size=200,
                    node_color=input,
                    font_color="white",
                    cmap = plt.cm.jet,
                    ax=axes[i, j],
                )
                plt.colorbar(color_map(input), ax=axes[i, j])
    plt.suptitle(title)
    plt.savefig(save_path)
    plt.close()


def plot_graph_grid_activations(
    node_colour,
    graph,
    save_path,
    title,
    edege_lables,
    number_graph_batch=0,
):
    nx_graph = convert_jraph_to_networkx_graph(graph, number_graph_batch)
    output = get_activations_graph_n(node_colour, graph, number_graph_batch=0)
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    pos = nx.spring_layout(nx_graph, iterations=100, seed=39775)
    ax.title.set_text(title)
    if edege_lables:
        labels = nx.get_edge_attributes(nx_graph, "edge_feature")
        nx.draw_networkx_edge_labels(nx_graph, pos=pos, edge_labels=labels, ax=ax)
    nx.draw(
        nx_graph,
        pos=pos,
        with_labels=True,
        node_size=500,
        node_color=output,
        font_color="white",
        ax=ax,
        cmap=plt.cm.jet,
    )
    plt.colorbar(color_map(output))
    plt.savefig(save_path)
    plt.close()


def plot_message_passing_layers_units(
    activations,
    targets,
    outputs,
    graph,
    number_hidden,
    n_message_passing,
    edege_lables,
    save_path,
    title,
):
    # minim 2 otherwise it breaks
    fig, axes = plt.subplots(n_message_passing, number_hidden)
    fig.set_size_inches(15, 15)
    nx_graph = convert_jraph_to_networkx_graph(graph, 0)
    pos = nx.spring_layout(nx_graph, iterations=100, seed=39775)
    if edege_lables:
        u = 0
        for k, m in nx_graph.edges():
            nx_graph[k][m]["weight"] = round(graph.edges[u][2], 2)
            u = u + 1
        labels = nx.get_edge_attributes(nx_graph, "weight")
    for i in range(n_message_passing):
        for j in range(number_hidden):
            activation = activations[i]
            axes[i, j].title.set_text(
                "first_graph_unit_" + str(j) + "message_passing_" + str(i)
            )
            # We select the first graph only
            input = get_activations_graph_n(activation.nodes[:, j].tolist(), graph, 0)
            nx.draw(
                nx_graph,
                pos=pos,
                with_labels=True,
                node_size=200,
                node_color=input,
                font_color="white",
                cmap=plt.cm.jet,
                ax=axes[i, j],
            )
            plt.colorbar(color_map(input), ax=axes[i, j])
            if edege_lables:
                nx.draw_networkx_edge_labels(
                    nx_graph, pos=pos, edge_labels=labels, ax=axes[i, j]
                )
    plt.suptitle(title)
    plt.savefig(save_path)
    plt.close()



def plot_curves(curves, path, title, legend_labels=None, x_label=None, y_label=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title)

    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)

    colors = ["b", "g", "r", "c", "m", "y", "k"]

    for i, curve in enumerate(curves):
        label = legend_labels[i] if legend_labels else None
        color = colors[i % len(colors)]
        ax.plot(curve, label=label, color=color)

    if legend_labels:
        ax.legend()
    plt.savefig(path)
    plt.show()
    plt.close()
