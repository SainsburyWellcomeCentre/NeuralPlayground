from .main_config import NPGConfig


class TrajectoryConfig(NPGConfig):
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


class RateMapConfig(NPGConfig):
    def __init__(self, **kwargs):
        self.FIGURE_SIZE = kwargs["figure_size"]
        self.RATEMAP_COLORMAP = kwargs["ratemap_colormap"]
        self.BIN_SIZE = kwargs["bin_size"]
        self.LABEL_FONTSIZE = kwargs["label_fontsize"]
        self.TICK_LABEL_FONTSIZE = kwargs["tick_label_fontsize"]
        self.COLORBAR_LABEL_FONTSIZE = kwargs["colorbar_label_fontsize"]


class PlotsConfig(NPGConfig):
    def __init__(self, plot_config: dict):
        self.TRAJECTORY = TrajectoryConfig(**plot_config["trajectory_plot"])
        self.RATEMAP = RateMapConfig(**plot_config["ratemap_plot"])
