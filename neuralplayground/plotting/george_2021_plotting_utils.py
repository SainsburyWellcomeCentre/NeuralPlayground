# CSCG: (FROM COLAB)
import math

import igraph
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.ndimage import gaussian_filter


def cscg_plot_graph(chmm, x, a, output_file, cmap=cm.Spectral, multiple_episodes=False, vertex_size=30, edge_threshold=0.1):
    """Plots the learned graph structure."""
    x = np.array(x, dtype=np.int64)
    a = np.array(a, dtype=np.int64)

    states = chmm.decode(x, a)[1]
    n_clones = chmm.n_clones
    v = np.unique(states)

    T = chmm.C[:, v][:, :, v]
    A = T.sum(0)
    row_sums = A.sum(1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    A /= row_sums

    A_thresholded = A.copy()
    A_thresholded[A_thresholded < edge_threshold] = 0

    g = igraph.Graph.Adjacency((A_thresholded > 0).tolist())

    obs_map = np.repeat(np.arange(len(n_clones)), n_clones)
    node_labels = obs_map[v]

    max_label = max(node_labels) if len(node_labels) > 0 else 1
    colors = [cmap(nl / max_label)[:3] for nl in node_labels]

    out = igraph.plot(
        g,
        output_file,
        layout=g.layout("kamada_kawai"),
        vertex_color=colors,
        vertex_label=v,
        vertex_size=vertex_size,
        margin=50,
        bbox=(600, 600),
        edge_width=0.5,
        edge_arrow_size=0.5,
    )
    return out


def plot_chmm_place_fields(agent, arena, plot_path=None):
    if hasattr(agent, "mess_fwd"):
        beliefs = agent.mess_fwd()
    else:
        beliefs = agent.get_belief_state()
    if len(beliefs) == 0:
        print("WARNING: mess_fwd returned empty array.")
        return None

    flat_positions = []
    for ep in agent.episode_history:
        flat_positions.extend(ep["pos"])
    if hasattr(agent, "current_episode") and len(agent.current_episode["pos"]) > 0:
        flat_positions.extend(agent.current_episode["pos"])
    if len(flat_positions) == 0:
        print("WARNING: No position data found.")
        return None

    min_len = min(len(beliefs), len(flat_positions))
    beliefs = beliefs[:min_len]
    flat_positions = flat_positions[:min_len]

    if hasattr(arena, "custom_layout") and arena.custom_layout is not None:
        h, w = arena.custom_layout.shape
    elif hasattr(arena, "room_layout"):
        h, w = arena.room_layout.shape
    else:
        h = int(arena.arena_y_limits[1] - arena.arena_y_limits[0])
        w = int(arena.arena_x_limits[1] - arena.arena_x_limits[0])

    grid_width = arena.resolution_w
    n_clones = beliefs.shape[1]

    activity_maps = np.zeros((n_clones, h, w))
    occupancy_map = np.zeros((h, w))

    for t, pos in enumerate(flat_positions):
        # pos is the full state: [flat_state_index, one_hot_vector, (x,y)]
        # pos[0] is the flat grid index recorded by pos_to_state() at step time
        flat_idx = int(pos[0])
        r = flat_idx // grid_width
        c = flat_idx % grid_width
        if 0 <= r < h and 0 <= c < w:
            activity_maps[:, r, c] += beliefs[t]
            occupancy_map[r, c] += 1

    occupancy_map[occupancy_map == 0] = 1e-8
    place_fields = activity_maps / occupancy_map[None, :, :]

    for i in range(n_clones):
        place_fields[i] = gaussian_filter(place_fields[i], sigma=0.0)

    n_cols = 10
    n_rows = math.ceil(n_clones / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = axes.flatten()

    for i in range(n_clones):
        axes[i].imshow(place_fields[i], origin="upper", cmap="viridis")
        axes[i].set_title(f"C {i}", fontsize=8)
        axes[i].axis("off")
    for i in range(n_clones, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Place fields saved to {plot_path}")
    return fig
