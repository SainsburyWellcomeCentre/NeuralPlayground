import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from neuralplayground.config import PLOT_CONFIG


def test_function():
    print(str(PLOT_CONFIG.TRAJECTORY))


def make_plot_trajectories(arena_limits, x, y, ax, plot_every, fontsize=24):
    """

    Parameters
    ----------
    x: ndarray (n_samples,)
        x position throughout recording of the given session
    y: ndarray (n_samples,)
        y position throughout recording of the given session
    ax: mpl.axes._subplots.AxesSubplot (matplotlib axis from subplots)
        axis from subplot from matplotlib where the ratemap will be plotted.
    plot_every: int
        time steps skipped to make the plot to reduce cluttering
    fontsize: int
        fontsize of labels in the plot

    Returns
    -------
    ax: mpl.axes._subplots.AxesSubplot (matplotlib axis from subplots)
        Modified axis where the trajectory is plotted
    """
    # Plotting borders of the arena
    config_vars = PLOT_CONFIG.TRAJECTORY
    ax.plot(
        [arena_limits[0, 0], arena_limits[0, 0]],
        [arena_limits[1, 0], arena_limits[1, 1]],
        config_vars.EXTERNAL_WALL_COLOR,
        lw=config_vars.EXTERNAL_WALL_THICKNESS,
    )
    ax.plot(
        [arena_limits[0, 1], arena_limits[0, 1]],
        [arena_limits[1, 0], arena_limits[1, 1]],
        config_vars.EXTERNAL_WALL_COLOR,
        lw=config_vars.EXTERNAL_WALL_THICKNESS,
    )
    ax.plot(
        [arena_limits[0, 0], arena_limits[0, 1]],
        [arena_limits[1, 1], arena_limits[1, 1]],
        config_vars.EXTERNAL_WALL_COLOR,
        lw=config_vars.EXTERNAL_WALL_THICKNESS,
    )
    ax.plot(
        [arena_limits[0, 0], arena_limits[0, 1]],
        [arena_limits[1, 0], arena_limits[1, 0]],
        config_vars.EXTERNAL_WALL_COLOR,
        lw=config_vars.EXTERNAL_WALL_THICKNESS,
    )

    # Setting colormap of trajectory
    cmap = mpl.cm.get_cmap("plasma")
    norm = plt.Normalize(0, np.size(x))

    aux_x = []
    aux_y = []
    for i in range(len(x)):
        if i % plot_every == 0:
            if i + plot_every >= len(x):
                break
            x_ = [x[i], x[i + plot_every]]
            y_ = [y[i], y[i + plot_every]]
            aux_x.append(x[i])
            aux_y.append(y[i])
            sc = ax.plot(x_, y_, "-", color=cmap(norm(i)), alpha=0.6)

    # Setting plot labels
    ax.set_xlabel("width", fontsize=fontsize)
    ax.set_ylabel("depth", fontsize=fontsize)
    ax.set_title("position", fontsize=fontsize)
    ax.grid(False)

    cmap = mpl.cm.get_cmap("plasma")
    norm = plt.Normalize(0, np.size(x))
    sc = ax.scatter(aux_x, aux_y, c=np.arange(len(aux_x)), vmin=0, vmax=len(x), cmap="plasma", alpha=0.6, s=0.1)

    # Setting colorbar to show number of sampled (time steps) recorded
    cbar = plt.colorbar(sc, ax=ax, ticks=[0, len(x)])
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.set_ylabel("N steps", rotation=270, fontsize=fontsize)
    cbar.ax.set_yticklabels([0, len(x)], fontsize=fontsize)
    ax.set_xlim([np.amin([x.min(), y.min()]) - 1.0, np.amax([x.max(), y.max()]) + 1.0])
    ax.set_ylim([np.amin([x.min(), y.min()]) - 1.0, np.amax([x.max(), y.max()]) + 1.0])
    return ax
