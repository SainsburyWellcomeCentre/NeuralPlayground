from .main_config import NPGConfig


class TrajectoryConfig(NPGConfig):
    """Config object for trajectory plots

    Attributes
    ----------
    FIGURE_SIZE: tuple
        Figure size
    TRAJECTORY_COLORMAP: str
        Colormap for trajectory plots
    TRAJECTORY_ALPHA: float
        Alpha value for the trajectory plot
    EXTERNAL_WALL_COLOR: str
        Color for the external wall of the arena
    EXTERNAL_WALL_THICKNESS: float
        Thickness of the external wall of the arena
    CUSTOM_WALL_COLOR: str
        Color for the custom wall of the arena
    CUSTOM_WALL_THICKNESS: float
        Thickness of the custom wall of the arena
    SCATTER_ALPHA: float
        Alpha value for the scatter plot
    SCATTER_MARKER: str
        Marker for the scatter plot
    SCATTER_MARKER_SIZE: float
        Size of the marker for the scatter plot
    LABEL_FONTSIZE: float
        Fontsize of labels in the plot
    TICK_LABEL_FONTSIZE: float
        Fontsize of tick labels in the plot
    GRID: bool
        Whether to show grid in the plot
    TITLE_FONTSIZE: float
        Fontsize of the title in the plot
    COLORBAR_LABEL_FONTSIZE: float
        Fontsize of the colorbar label in the plot
    """

    def __init__(self, **kwargs):
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
        self.RATEMAP_COLORMAP = kwargs["ratemap_colormap"]
        self.BIN_SIZE = kwargs["bin_size"]
        self.LABEL_FONTSIZE = kwargs["label_fontsize"]
        self.TICK_LABEL_FONTSIZE = kwargs["tick_label_fontsize"]
        self.TITLE_FONTSIZE = kwargs["title_fontsize"]
        self.COLORBAR_LABEL_FONTSIZE = kwargs["colorbar_label_fontsize"]
        self.GRID = kwargs["grid"]


class AgentCompatisonConfig(NPGConfig):
    """Config object for ratemap plots

    Attributes
    ----------

    TABLE_FONTSIZE: float
        Fontsize of table


    """

    def __init__(self, **kwargs):
        self.FIGSIZE = kwargs["figsize"]
        self.FONTSIZE = kwargs["fontsize"]
        self.PLOT_SAC_EXP = kwargs["plot_sac_exp"]
        self.PLOT_SAC_AGT = kwargs["plot_sac_agt"]


class TableConfig(NPGConfig):
    """Config object for ratemap plots

    Attributes
    ----------
    ROW_HEIGHT: float
        Height of rows in the table
    COL_WIDTH: float
        Width of columns in the table
    TABLE_FONTSIZE: float
        Fontsize of table
    ROW_COLOR: str
        Color of rows in the table
    HEADER_COLOR: str
        Color of header in the table
    EDGE_COLOR: str
        Color of edges in the table
    HEADER_COLLUMNS: list
        List of header columns in the table
    BBOX: tuple
        Bounding box of the table
    """

    def __init__(self, **kwargs):
        self.ROW_HEIGHT = kwargs["row_height"]
        self.COL_WIDTH = kwargs["col_width"]
        self.TABLE_FONTSIZE = kwargs["table_fontsize"]
        self.ROW_COLOR = kwargs["row_colors"]
        self.HEADER_COLOR = kwargs["header_color"]
        self.TEXT_COLOR = kwargs["text_color"]
        self.EDGE_COLOR = kwargs["edge_color"]
        self.HEADER_COLUMNS = kwargs["header_columns"]
        self.BBOX = kwargs["bbox"]


class PlotsConfig(NPGConfig):

    """Config object for plots, all plots are config are stored in this object
    Attributes
    ----------
    TRAJECTORY: TrajectoryConfig
        Config object for trajectory plots
    RATEMAP: RateMapConfig
        Config object for ratemap plots
    AGENT_COMPARISON: AgentCompatisonConfig
        Config object for agent comparison plots
    TABLE: TableConfig
        Config object for table plots
    """

    def __init__(self, plot_config: dict):
        self.TRAJECTORY = TrajectoryConfig(**plot_config["trajectory_plot"])
        self.RATEMAP = RateMapConfig(**plot_config["ratemap_plot"])
        self.AGENT_COMPARISON = AgentCompatisonConfig(**plot_config["agent_comparison_plot"])
        self.TABLE = TableConfig(**plot_config["table_plot"])
