# @title Make rng sequence generator
import matplotlib.pyplot as plt
import networkx as nx
from class_utils import convert_jraph_to_networkx_graph,get_activations_graph_n,get_node_pad


def plot_input_target_output(inputs, targets, outputs, graph, n, edege_lables, save_path):
    # minim 2 otherwise it breaks
    rows = ["{}".format(row) for row in ["Input", "Target", "Outputs"]]
    fig, axes = plt.subplots(3, n)
    fig.set_size_inches(10, 10)
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
            for l, j in nx_graph.edges():
                nx_graph[l][j]["weight"] = round(graph.edges[node_padd + u][2], 2)
                u = u + 1
            labels = nx.get_edge_attributes(nx_graph, "weight")
            nx.draw_networkx_edge_labels(nx_graph, pos=pos, edge_labels=labels, ax=axes[0, i])
            nx.draw_networkx_edge_labels(nx_graph, pos=pos, edge_labels=labels, ax=axes[1, i])
            nx.draw_networkx_edge_labels(nx_graph, pos=pos, edge_labels=labels, ax=axes[2, i])
            #            labels = nx.get_edge_attributes(nx_graph, 'weight')

        nx.draw(nx_graph, pos=pos, with_labels=True, node_size=200, node_color=input, font_color="white", ax=axes[0, i])
        nx.draw(nx_graph, pos=pos, with_labels=True, node_size=200, node_color=target, font_color="white", ax=axes[1, i])

        nx.draw(nx_graph, pos=pos, with_labels=True, node_size=200, node_color=output, font_color="white", ax=axes[2, i])

    for axes, row in zip(axes[:, 0], rows):
        axes.set_ylabel(row, rotation=0, size="large")
    plt.savefig(save_path)


def plot_message_passing_layers(inputs, activations, targets, outputs, graph, n, n_message_passing, edege_lables, save_path):
    # minim 2 otherwise it breaks
    fig, axes = plt.subplots(n_message_passing + 3, n)
    fig.set_size_inches(10, 10)
    for j in range(n):
        nx_graph = convert_jraph_to_networkx_graph(graph, j)
        pos = nx.spring_layout(nx_graph, iterations=100, seed=39775)
        if edege_lables:
            node_padd = get_node_pad(graph, j)
            u = 0
            for k, m in nx_graph.edges():
                nx_graph[k][m]["weight"] = round(graph.edges[node_padd + u][2], 2)
                u = u + 1
            labels = nx.get_edge_attributes(nx_graph, "weight")
        for i in range(n_message_passing + 3):
            if edege_lables:
                nx.draw_networkx_edge_labels(nx_graph, pos=pos, edge_labels=labels, ax=axes[i, j])
            if i == (n_message_passing + 2):
                axes[i, j].title.set_text("input")
                input = get_activations_graph_n(inputs, graph, j)
                nx.draw(
                    nx_graph, pos=pos, with_labels=True, node_size=200, node_color=input, font_color="white", ax=axes[i, j]
                )
            elif i == (n_message_passing + 1):
                axes[i, j].title.set_text("target")
                target = get_activations_graph_n(targets, graph, j)
                nx.draw(
                    nx_graph, pos=pos, with_labels=True, node_size=200, node_color=target, font_color="white", ax=axes[i, j]
                )

            elif i == (n_message_passing):
                axes[i, j].title.set_text("output")
                output = get_activations_graph_n(outputs, graph, j)
                nx.draw(
                    nx_graph, pos=pos, with_labels=True, node_size=200, node_color=output, font_color="white", ax=axes[i, j]
                )
            else:
                activation = activations[i]
                axes[i, j].title.set_text("graph_" + str(j) + "message_passing_" + str(i))
                input = get_activations_graph_n(activation.nodes[:, j].tolist(), graph, j)
                nx.draw(
                    nx_graph, pos=pos, with_labels=True, node_size=200, node_color=input, font_color="white", ax=axes[i, j]
                )

    plt.savefig(save_path)


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
        node_padd = get_node_pad(graph, number_graph_batch)
        u = 0
        for k, m in nx_graph.edges():
            nx_graph[k][m]["weight"] = round(graph.edges[node_padd + u][2], 2)
            u = u + 1
        labels = nx.get_edge_attributes(nx_graph, "weight")
        nx.draw_networkx_edge_labels(nx_graph, pos=pos, edge_labels=labels, ax=ax)
    nx.draw(nx_graph, pos=pos, with_labels=True, node_size=500, node_color=output, font_color="white", ax=ax)
    plt.savefig(save_path)


def plot_message_passing_layers_units(
    activations, targets, outputs, graph, number_hidden, n_message_passing, edege_lables, save_path
):
    # minim 2 otherwise it breaks
    fig, axes = plt.subplots(n_message_passing, number_hidden)
    fig.set_size_inches(12, 12)
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
            axes[i, j].title.set_text("first_graph_unit_" + str(j) + "message_passing_" + str(i))
            # We select the first graph only
            input = get_activations_graph_n(activation.nodes[:, j].tolist(), graph, 0)
            nx.draw(nx_graph, pos=pos, with_labels=True, node_size=200, node_color=input, font_color="white", ax=axes[i, j])
            if edege_lables:
                nx.draw_networkx_edge_labels(nx_graph, pos=pos, edge_labels=labels, ax=axes[i, j])
    plt.savefig(save_path)



def plot_xy(auc_roc, path, title):
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.title.set_text(title)
    ax.plot(auc_roc)
    plt.savefig(path)
