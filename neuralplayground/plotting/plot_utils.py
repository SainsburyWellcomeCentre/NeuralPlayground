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
        y axis label, "depth" by default when called
    title_x: str
        x axis label, "width" by default when called
    title_cbar: str
        colorbar label, "spikes per bin" by default when called

    Returns
    -------
    ax: mpl.axes._subplots.AxesSubplot (matplotlib axis from subplots)
        Modified axis where ratemap is plotted
    """

    # Formating ratemap plot
    config_vars = PLOT_CONFIG.RATEMAP
    sc = ax.imshow(h, cmap=config_vars.RATEMAP_COLORMAP)
    cbar = plt.colorbar(sc, ax=ax, ticks=[np.min(h), np.max(h)], orientation="horizontal", fraction=0.046)
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
    return ax


def render_mpl_table(data, ax=None, **kwargs):
    """https://stackoverflow.com/questions/19726663/how-to-save-the-pandas-dataframe-series-data-as-a-figure

    Render an image of a table contained in a pandas dataframe.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to render.

    col_width : float, optional
        Width of columns in table.

    row_height : float, optional
        Height of rows in table.

    font_size : int, optional
        Font size in table.

    bbox : list, optional
        Bounding box (coordinates) for the table.

    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axis to render table in.

    kwargs : dict, optional
        Dictionary of extra keyword arguments to pass to
        :meth:`matplotlib.table.Table`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the table.

    ax : matplotlib.axes._subplots.AxesSubplot
        Axis the table was rendered in.
    """

    config_vars = PLOT_CONFIG.TABLE
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([config_vars.col_width, config_vars.row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis("off")
    ax.set_axis_off()
    mpl_table = ax.table(cellText=data.values, bbox=config_vars.BBOX, colLabels=data.columns, **kwargs, cellLoc="center")
    mpl_table.auto_set_font_size(False)
    mpl_table.auto_set_column_width(col=list(range(len(data.columns))))
    mpl_table.set_fontsize(config_vars.TABLE_FONTSIZE)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(config_vars.EDGE_COLOR)
        if k[0] == 0 or k[1] < config_vars.HEADER_COLUMNS:
            cell.set_text_props(weight="bold", color=config_vars.TEXT_COLOR)
            cell.set_facecolor(config_vars.HEADER_COLOR)
        else:
            cell.set_facecolor(config_vars.ROW_COLOR[k[0] % len(config_vars.ROW_COLOR)])
    return ax.get_figure(), ax


def make_agent_comparison(envs, parameters, agents, exps=None, recording_index=None, tetrode_id=None, GridScorer=None):
    """Plot function to compare agents in a given environment

    Parameters
    ----------
    env : object of class Environment
    parameters :
    agents: list of objects of class Agent
    exp: object of class Experiment
    ax : matplotlib axis

    Returns
    -------
    ax: matplotlib axis
    """
    config_vars = PLOT_CONFIG.AGENT_COMPARISON
    exp_data = False
    for j, env in enumerate(envs):
        if hasattr(env, "show_data"):
            exp_data = True

    if exps is not None:
        if exp_data:
            f, ax = plt.subplots(
                5,
                len(agents) + len(envs) + len(exps),
                figsize=(config_vars.FIGSIZE[0] * (len(agents) + 2), config_vars.FIGSIZE[1]),
            )
        else:
            f, ax = plt.subplots(
                3,
                len(agents) + len(envs) + len(exps),
                figsize=(config_vars.FIGSIZE[0] * (len(agents) + 1), config_vars.FIGSIZE[1]),
            )
    else:
        if exp_data:
            f, ax = plt.subplots(
                5, len(agents) + len(envs), figsize=(config_vars.FIGSIZE[0] * (len(agents) + 1), config_vars.FIGSIZE[1])
            )
        else:
            f, ax = plt.subplots(
                3, len(agents) + len(envs), figsize=(config_vars.FIGSIZE[0] * (len(agents) + 1), config_vars.FIGSIZE[1])
            )

    for k, env in enumerate(envs):
        env.plot_trajectory(ax=ax[1, k])
        ax[2, k].set_axis_off()
        # render_mpl_table( pd.DataFrame([parameters[0]["env_params"]]),ax=ax[0, k],)

        ax[0, k].text(0, 1.1, env.environment_name, fontsize=config_vars.FONTSIZE)
        ax[0, k].set_axis_off()
        for p, text in enumerate(parameters[k]["env_params"]):
            ax[0, k].text(0, 1, "Event param", fontsize=10)
            variable = parameters[k]["env_params"][text]
            ax[0, k].text(0, 0.9 - ((p) * 0.1), text + ": " + str(variable), fontsize=10)
            ax[0, k].set_axis_off()

        if hasattr(env, "show_data"):
            ax[2, k].set_axis_off()
            render_mpl_table(
                data=env.recording_list,
                ax=ax[2, k],
            )
            env.plot_recording_tetr(recording_index=recording_index, tetrode_id=tetrode_id, ax=ax[3][k])
            r_out_im, x_bin, y_bin = env.recording_tetr()
            GridScorer_SR = GridScorer(x_bin - 1)
            GridScorer_SR.plot_grid_score(r_out_im=r_out_im, plot=config_vars.PLOT_SAC_EXP, ax=ax[4][k])
        else:
            if exp_data:
                ax[2][k].set_axis_off()
                ax[3][k].set_axis_off()
                ax[4][k].set_axis_off()
            else:
                ax[2][k].set_axis_off()

    for i, agent in enumerate(agents):
        if hasattr(agent, "plot_rate_map"):
            agent.plot_rate_map(ax=ax[1][1 + i + k])
            GridScorer_SR = GridScorer(agent.resolution_width)
            GridScorer_SR.plot_grid_score(
                r_out_im=agent.get_rate_map_matrix(), plot=config_vars.PLOT_SAC_AGT, ax=ax[2, 1 + i + k]
            )
            for j, text in enumerate(parameters[i]["agent_params"]):
                if j > 9:
                    variable = parameters[i]["agent_params"][text]
                    ax[0, i + k + 1].text(0.7, 1 - ((j - 9) * 0.1), text + ": " + str(variable), fontsize=10)
                    ax[0, i + k + 1].set_axis_off()
                else:
                    ax[0, i + k + 1].text(0, 1, "Agent param", fontsize=10)
                    variable = parameters[i]["agent_params"][text]
                    ax[0, i + k + 1].text(0, 0.9 - ((j) * 0.1), text + ": " + str(variable), fontsize=10)
                    ax[0, i + k + 1].set_axis_off()

            # render_mpl_table(data=pd.DataFrame([parameters[i]["agent_params"]]), ax=ax[0, 1 + i + k])
        if exp_data:
            ax[3][1 + i + k].set_axis_off()
            ax[4][1 + i + k].set_axis_off()

    if exps is not None:
        for m, exp in enumerate(exps):
            if exp is not None:
                ax[0, i + k + m + 2].text(0, 1.1, exp.experiment_name, fontsize=config_vars.FONTSIZE)

                render_mpl_table(exp.recording_list, ax=ax[0, i + k + m + 2])
                exp.plot_recording_tetr(recording_index=recording_index, tetrode_id=tetrode_id, ax=ax[1][i + k + m + 2])
                r_out_im, x_bin, y_bin = exp.recording_tetr()
                GridScorer_SR = GridScorer(x_bin - 1)
                GridScorer_SR.plot_grid_score(r_out_im=r_out_im, plot=config_vars.PLOT_SAC_EXP, ax=ax[2][i + k + m + 2])
                if exp_data:
                    ax[3][i + k + m + 2].set_axis_off()
                    ax[4][i + k + m + 2].set_axis_off()
    return ax
