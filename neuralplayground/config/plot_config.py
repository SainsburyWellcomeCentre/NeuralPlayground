from .main_config import NPGConfig


class TrajectoryConfig(NPGConfig):
    """Config object for trajectory plots

    Attributes
    ----------
    FIGURE_SIZE: tuple
        Figure size
    TRAJECTORY_COLORMAP: str
        Colormap for agent trajectory plots
    TRAJECTORY_ALPHA: float
        Alpha value for agent trajectory plots
    EXTERNAL_WALL_COLOR: str
        Color of external walls
    EXTERNAL_WALL_THICKNESS: float
        Thickness of external walls
    CUSTOM_WALL_COLOR: str
        Color of custom walls
    CUSTOM_WALL_THICKNESS: float
        Thickness of custom walls
    SCATTER_ALPHA: float
        Alpha value for scatter dots in the agent trajectory plot
    SCATTER_MARKER: str
        Marker for scatter dots in the agent trajectory plot
    SCATTER_MARKER_SIZE: float
        Marker size for scatter dots in the agent trajectory plot
    LABEL_FONTSIZE: float
        Fontsize of labels in the plot
    TICK_LABEL_FONTSIZE: float
        Fontsize of tick labels in the plot
    PLOT_EVERY_POINTS: int
        Time steps skipped to make the plot to reduce cluttering
    GRID: bool
        Boolean value to plot grid in the background of trajectory plots
    """

    def __init__(self, **kwargs):
        self.FIGURE_SIZE = kwargs["figure_size"]
        self.TRAJECTORY_COLORMAP = kwargs["trajectory_colormap"]
        self.TRAJECTORY_ALPHA = kwargs["trajectory_alpha"]
        self.EXTERNAL_WALL_COLOR = kwargs["external_wall_color"]
        self.EXTERNAL_WALL_THICKNESS = kwargs["external_wall_thickness"]
        self.CUSTOM_WALL_COLOR = kwargs["custom_wall_color"]
        self.CUSTOM_WALL_THICKNESS = kwargs["custom_wall_thickness"]
        self.SCATTER_ALPHA = kwargs["scatter_alpha"]
        self.SCATTER_MARKER = kwargs["scatter_marker"]
        self.SCATTER_MARKER_SIZE = kwargs["scatter_marker_size"]
        self.LABEL_FONTSIZE = kwargs["label_fontsize"]
        self.TICK_LABEL_FONTSIZE = kwargs["tick_label_fontsize"]
        self.PLOT_EVERY_POINTS = kwargs["plot_every_points"]
        self.GRID = kwargs["grid"]
        self.TITLE_FONTSIZE = kwargs["title_fontsize"]
        self.COLORBAR_LABEL_FONTSIZE = kwargs["colorbar_label_fontsize"]


class RateMapConfig(NPGConfig):
    """Config object for ratemap plots

    Attributes
    ----------
    FIGURE_SIZE: tuple
        Figure size
    RATEMAP_COLORMAP: str
        Colormap for ratemap plots
    BIN_SIZE: float
        Size of bins in 2D space to generate the ratemap
    LABEL_FONTSIZE: float
        Fontsize of labels in the plot
    TICK_LABEL_FONTSIZE: float
        Fontsize of tick labels in the plot
    COLORBAR_LABEL_FONTSIZE: float
        Fontsize of colorbar labels in the plot
    """

    def __init__(self, **kwargs):
        self.FIGURE_SIZE = kwargs["figure_size"]
        self.RATEMAP_COLORMAP = kwargs["ratemap_colormap"]
        self.BIN_SIZE = kwargs["bin_size"]
        self.LABEL_FONTSIZE = kwargs["label_fontsize"]
        self.TICK_LABEL_FONTSIZE = kwargs["tick_label_fontsize"]
        self.TITLE_FONTSIZE = kwargs["title_fontsize"]
        self.COLORBAR_LABEL_FONTSIZE = kwargs["colorbar_label_fontsize"]


class PlotsConfig(NPGConfig):
    """Config object for plots, all plots are config are stored in this object"""

    def __init__(self, plot_config: dict):
        self.TRAJECTORY = TrajectoryConfig(**plot_config["trajectory_plot"])
        self.RATEMAP = RateMapConfig(**plot_config["ratemap_plot"])
