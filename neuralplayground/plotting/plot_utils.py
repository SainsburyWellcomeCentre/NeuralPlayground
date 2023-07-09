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
    lower_lim, upper_lim = np.amin(arena_limits), np.amax(arena_limits)
    ax.set_xlim([lower_lim, upper_lim])
    ax.set_ylim([lower_lim, upper_lim])
    return ax


def make_plot_rate_map(h, ax, title, title_x, title_y, title_cbar):
    """plot function with formating of ratemap plot

    Parameters
    ----------
    h: ndarray (nybins, nxbins)
        Number of spikes falling on each bin through the recorded session, nybins number of bins in y axis,
        nxbins number of bins in x axis
    ax: mpl.axes._subplots.AxesSubplot (matplotlib axis from subplots)
        axis from subplot from matplotlib where the ratemap will be plotted.
    title: str
        plot title, tetrode id by default when called
    save_path: str, list of str, tuple of str
        saving path of the generated figure, if None, no figure is saved

    Returns
    -------
    ax: mpl.axes._subplots.AxesSubplot (matplotlib axis from subplots)
        Modified axis where ratemap is plotted
    """

    # Formating ratemap plot
    sc = ax.imshow(h, cmap="jet")
    cbar = plt.colorbar(sc, ax=ax, ticks=[np.min(h), np.max(h)], orientation="horizontal")
    cbar.ax.set_xlabel(title_cbar, fontsize=12)
    cbar.ax.set_xticklabels([np.round(np.min(h)), np.round(np.max(h))], fontsize=12)
    ax.set_title(title)
    ax.set_ylabel(title_y, fontsize=16)
    ax.set_xlabel(title_x, fontsize=16)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # Save if save_path is not None
    return ax
