from .main_config import NPGConfig


class TrajectoryConfig(NPGConfig):
    def __init__(self):
        self.FIGURE_SIZE = (8, 6)
        self.TRAJECTORY_COLORMAP = "plasma"
        self.TRAJECTORY_ALPHA = 0.6
        self.EXTERNAL_WALL_COLOR = "C3"
        self.EXTERNAL_WALL_THICKNESS = 3
        self.CUSTOM_WALL_COLOR = "C0"
        self.CUSTOM_WALL_THICKNESS = 3
        self.SCATTER_ALPHA = 0.6
        self.SCATTER_MARKER = "o"
        self.SCATTER_MARKER_SIZE = 0.1
        self.LABEL_FONTSIZE = 24
        self.TICK_LABEL_FONTSIZE = 12
        self.PLOT_EVERY_POINTS = 20
        self.GRID = True


class RateMapConfig(NPGConfig):
    def __init__(self):
        self.FIGURE_SIZE = (8, 6)
        self.RATEMAP_COLORMAP = "jet"
        self.BIN_SIZE = 2.0
        self.LABEL_FONTSIZE = 24
        self.TICK_LABEL_FONTSIZE = 12
        self.COLORBAR_LABEL_FONTSIZE = 12


class PlotsConfig(NPGConfig):
    def __init__(self):
        self.TRAJECTORY = TrajectoryConfig()
        self.RATEMAP = RateMapConfig()
