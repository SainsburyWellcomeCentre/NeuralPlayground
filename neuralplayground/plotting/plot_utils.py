import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from neuralplayground.config import PLOT_CONFIG


def make_plot_trajectories(arena_limits, x, y, ax, plot_every):
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
    # PLOT_CONFIG.TRAJECTORY.TRAJECTORY_COLORMAP = "bla"
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
    cmap = mpl.cm.get_cmap(config_vars.TRAJECTORY_COLORMAP)
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
    ax.set_xlabel("width", fontsize=config_vars.LABEL_FONTSIZE)
    ax.set_ylabel("depth", fontsize=config_vars.LABEL_FONTSIZE)
    ax.set_title("position", fontsize=config_vars.TITLE_FONTSIZE)
    ax.grid(config_vars.GRID)

    sc = ax.scatter(
        aux_x, aux_y, c=np.arange(len(aux_x)), vmin=0, vmax=len(x), cmap=config_vars.TRAJECTORY_COLORMAP, alpha=0.6, s=0.1
    )

    # Setting colorbar to show number of sampled (time steps) recorded
    cbar = plt.colorbar(sc, ax=ax, ticks=[0, len(x)])
    cbar.ax.tick_params(labelsize=config_vars.TICK_LABEL_FONTSIZE)
    cbar.ax.set_yticklabels([0, len(x)], fontsize=config_vars.COLORBAR_LABEL_FONTSIZE)
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
    title_y:  str
    Returns
    -------
    ax: mpl.axes._subplots.AxesSubplot (matplotlib axis from subplots)
        Modified axis where ratemap is plotted
    """

    # Formating ratemap plot

    config_vars = PLOT_CONFIG.RATEMAP
    sc = ax.imshow(h, cmap=config_vars.RATEMAP_COLORMAP)
    cbar = plt.colorbar(sc, ax=ax, ticks=[np.min(h), np.max(h)], orientation="horizontal")
    cbar.ax.set_xlabel(title_cbar, fontsize=config_vars.COLORBAR_LABEL_FONTSIZE)
    cbar.ax.set_xticklabels(
        [np.round(np.min(h), decimals=2), np.round(np.max(h), decimals=2)], fontsize=config_vars.TICK_LABEL_FONTSIZE
    )
    ax.set_title(title, fontsize=config_vars.TITLE_FONTSIZE)
    ax.set_ylabel(title_y, fontsize=config_vars.LABEL_FONTSIZE)
    ax.set_xlabel(title_x, fontsize=config_vars.LABEL_FONTSIZE)
    ax.grid(config_vars.GRID)
    ax.set_xticks([])
    ax.set_yticks([])
    # Save if save_path is not None
    return ax


def make_agent_comparison(env, parameters, agents, exp=None, ax=None):
    if exp is not None or not hasattr(env, "show_data"):
        f, ax = plt.subplots(2, len(agents) + 2, figsize=(8 * (len(agents) + 2), 7))
    else:
        f, ax = plt.subplots(2, len(agents) + 1, figsize=(8 * (len(agents) + 1), 7))
    env.plot_trajectory(ax=ax[1, 0])
    ax[0, 0].set_axis_off()
    for i, text in enumerate(parameters[0]["env_params"]):
        ax[0, 0].text(0, 1, "Env param", fontsize=10)
        variable = parameters[0]["env_params"][text]
        ax[0, 0].text(0, 0.9 - (i * 0.1), text + ": " + str(variable), fontsize=10)
    for i, agent in enumerate(agents):
        agent.plot_rates(ax=ax[1][i + 1])
        for k, text in enumerate(parameters[i]["agent_params"]):
            if k > 9:
                variable = parameters[i]["agent_params"][text]
                ax[0, i + 1].text(0.7, 1 - ((k - 9) * 0.1), text + ": " + str(variable), fontsize=10)
                ax[0, i + 1].set_axis_off()
            else:
                ax[0, i + 1].text(0, 1, "Agent param", fontsize=10)
                variable = parameters[i]["agent_params"][text]
                ax[0, i + 1].text(0, 0.9 - ((k) * 0.1), text + ": " + str(variable), fontsize=10)
                ax[0, i + 1].set_axis_off()
    if hasattr(env, "show_data"):
        ax[0, i + 2].set_axis_off()
        env.plot_recording_tetr(ax=ax[1][i + 2])
    if exp is not None:
        ax[0, i + 2].set_axis_off()
        exp.plot_recording_tetr(ax=ax[1][i + 2])
