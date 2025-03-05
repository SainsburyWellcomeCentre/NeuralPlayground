import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from neuralplayground.arenas.arena_core import Environment
from neuralplayground.plotting.plot_utils import make_plot_trajectories
from neuralplayground.utils import check_crossing_wall


class DiscreteObjectEnvironment(Environment):
    """
    Arena class which accounts for discrete sensory objects, inherits from the Simple2D class.

    Now allows two grid layouts:
        1) 'square' (original behaviour)
        2) 'hex' (hexagonal tiling)

    The parameter grid_type='square' or 'hex' can be passed in env_kwargs.
    """

    def __init__(
        self,
        recording_index: int = None,
        environment_name: str = "DiscreteObject",
        verbose: bool = False,
        experiment_class: str = None,
        **env_kwargs,
    ):
        """
        Initialize the class. env_kwargs arguments are specific for each of the child environments and
        described in their respective class.

        Parameters
        ----------
        grid_type : str in {'square', 'hex'}, optional
            If 'hex', positions are arranged in a hex tiling and the default walls are also a large hex.
            Otherwise 'square' is used.
        """
        super().__init__(environment_name, **env_kwargs)
        self.environment_name = environment_name
        self.use_behavioural_data = env_kwargs["use_behavioural_data"]
        self.grid_type = env_kwargs.get("grid_type", "square")  # NEW

        self.experiment = experiment_class(
            experiment_name=self.environment_name,
            data_path=env_kwargs["data_path"],
            recording_index=recording_index,
            verbose=verbose,
        )
        if self.use_behavioral_data:
            self.state_dims_labels = ["x_pos", "y_pos", "head_direction_x", "head_direction_y"]
            self.arena_limits = self.experiment.arena_limits
            self.arena_x_limits = self.arena_limits[0].astype(int)
            self.arena_y_limits = self.arena_limits[1].astype(int)
        else:
            self.state_dims_labels = ["x_pos", "y_pos"]
            self.arena_x_limits = env_kwargs["arena_x_limits"]
            self.arena_y_limits = env_kwargs["arena_y_limits"]

        self.n_objects = env_kwargs["n_objects"]
        self.state_density = env_kwargs["state_density"]
        self.arena_limits = np.array(
            [[self.arena_x_limits[0], self.arena_x_limits[1]], [self.arena_y_limits[0], self.arena_y_limits[1]]]
        )
        self.room_width = np.diff(self.arena_x_limits)[0]
        self.room_depth = np.diff(self.arena_y_limits)[0]
        self.agent_step_size = env_kwargs["agent_step_size"]
        self._create_default_walls()  # <--- Will pick square or hex below
        self._create_custom_walls()
        self.wall_list = self.default_walls + self.custom_walls

        # Create our grid coordinates (square or hex)
        self._create_grid()

        # Number of discrete states
        self.n_states = len(self.xy_combination)
        self.objects = self.generate_objects()

        # Keep the occupancy grid with the same shape for consistency
        self.occupancy_grid = np.zeros((int(self.room_depth * self.state_density), int(self.room_width * self.state_density)))

        self.steps_in_curr_env = 0
        self.max_steps_per_env = 5000

    def _create_grid(self):
        """
        Create the coordinate grid of discrete states, depending on self.grid_type.
        If 'square': create the same x_array, y_array as before.
        If 'hex': create a hex tiling in the same bounding box.
        """
        self.resolution_w = int(self.room_width * self.state_density)
        self.resolution_d = int(self.room_depth * self.state_density)
        self.state_size = 1 / self.state_density

        if self.grid_type == "square":
            # Original approach
            self.x_array = np.linspace(
                self.arena_x_limits[0] + self.state_size / 2,
                self.arena_x_limits[1] - self.state_size / 2,
                self.resolution_w,
            )
            self.y_array = np.linspace(
                self.arena_y_limits[0] + self.state_size / 2,
                self.arena_y_limits[1] - self.state_size / 2,
                self.resolution_d,
            )
            mesh = np.meshgrid(self.x_array, self.y_array)
            self.xy_combination = np.column_stack([mesh[0].ravel(), mesh[1].ravel()])

        else:  # self.grid_type == "hex"
            coords_list = []
            dx = self.state_size
            dy = self.state_size * np.sqrt(3) / 2
            for row in range(self.resolution_d):
                for col in range(self.resolution_w):
                    # Offset every other row by dx/2
                    offset_x = (col + 0.5 * (row % 2)) * dx + self.arena_x_limits[0] + dx / 2
                    offset_y = row * dy + self.arena_y_limits[0] + dy / 2
                    coords_list.append([offset_x, offset_y])
            self.xy_combination = np.array(coords_list)
            self.x_array, self.y_array = None, None

    def _create_default_walls(self):
        """
        If grid_type='square', keep the original rectangular walls.
        If grid_type='hex', create a perfectly symmetrical hex about (0,0)
        so that it takes exactly 5 unit steps from (0,0) to the boundary.
        """
        if self.grid_type == "square":
            # Original rectangular boundary
            self.default_walls = []
            self.default_walls.append(
                np.array(
                    [
                        [self.arena_limits[0, 0] - 0.1, self.arena_limits[1, 0] + 0.1],
                        [self.arena_limits[0, 0] - 0.1, self.arena_limits[1, 1] + 0.1],
                    ]
                )
            )
            self.default_walls.append(
                np.array(
                    [
                        [self.arena_limits[0, 1] + 0.1, self.arena_limits[1, 0] - 0.1],
                        [self.arena_limits[0, 1] + 0.1, self.arena_limits[1, 1] + 0.1],
                    ]
                )
            )
            self.default_walls.append(
                np.array(
                    [
                        [self.arena_limits[0, 0] - 0.1, self.arena_limits[1, 0] - 0.1],
                        [self.arena_limits[0, 1] + 0.1, self.arena_limits[1, 0] - 0.1],
                    ]
                )
            )
            self.default_walls.append(
                np.array(
                    [
                        [self.arena_limits[0, 0] - 0.1, self.arena_limits[1, 1] + 0.1],
                        [self.arena_limits[0, 1] + 0.1, self.arena_limits[1, 1] + 0.1],
                    ]
                )
            )
        else:
            # Hexagonal boundary
            # We assume arena_x_limits = [-5, 5], arena_y_limits = [-5, 5].
            # Centre is therefore (0,0), and we want radius ~5 so that it takes
            # exactly 5 steps (size=1) to reach the wall from the centre.

            self.default_walls = []

            # Tiny epsilon so floating-point rounding doesn't let the agent slip outside.
            # You can tweak e.g. 1e-4, 1e-3, etc. as needed.
            r = 5.3

            # For a 'pointy-topped' orientation, corners at angles: 0, 60, 120, ..., 300
            angles_deg = [0, 60, 120, 180, 240, 300]
            corners = []
            for deg in angles_deg:
                rad = np.radians(deg)
                x = r * np.cos(rad)
                y = r * np.sin(rad)
                corners.append([x, y])

            # Make walls from each corner to the next
            for i in range(6):
                j = (i + 1) % 6
                corner_i = np.array(corners[i])
                corner_j = np.array(corners[j])
                self.default_walls.append(np.vstack([corner_i, corner_j]))

    def _create_custom_walls(self):
        """Custom walls method. In this case is empty since the environment is a simple shape (square or hex)."""
        self.custom_walls = []

    def reset(self, random_state=True, custom_state=None):
        self.global_steps = 0
        self.global_time = 0
        self.history = []
        self.state = [-1, -1, [self.room_width + 1, self.room_depth + 1]]
        if random_state:
            pos = [
                np.random.uniform(low=self.arena_limits[0, 0], high=self.arena_limits[0, 1]),
                np.random.uniform(low=self.arena_limits[1, 0], high=self.arena_limits[1, 1]),
            ]
        else:
            pos = np.array([0, 0])

        if custom_state is not None:
            pos = np.array(custom_state)

        if self.use_behavioural_data:
            pos, head_dir = self.experiment.position[0, :], self.experiment.head_direction[0, :]
            custom_state = np.concatenate([pos, head_dir])

        self.objects = self.generate_objects()
        self.occupancy_grid = np.zeros((int(self.room_depth * self.state_density), int(self.room_width * self.state_density)))

        nearest_idx = np.argmin(np.sum((self.xy_combination - pos) ** 2, axis=1))
        snapped_pos = self.xy_combination[nearest_idx]
        observation = self.make_object_observation(snapped_pos)
        self.state = observation
        return observation, self.state

    def step(self, action: np.ndarray, normalize_step: bool = True, skip_every: int = 10):
        self.old_state = self.state.copy()
        if not self.use_behavioural_data:
            if normalize_step:
                new_pos_state = np.add(self.state[-1], [self.agent_step_size * e for e in action]).tolist()
            else:
                new_pos_state = np.add(self.state[-1], action).tolist()
            new_pos_state, invalid_action = self.validate_action(
                self.state[-1], [self.agent_step_size * e for e in action], new_pos_state[:2]
            )
        else:
            # If using behavioural data, handle that as in the original
            pass

        reward = self.reward_function(action, self.state[-1])
        observation = self.make_object_observation(new_pos_state)

        state_index = self.pos_to_state(new_pos_state)
        # Keep original occupancy grid usage
        w = int(self.room_width * self.state_density)
        self.occupancy_grid[state_index // w, state_index % w] += 1

        self.state = observation
        self.transition = {
            "action": action,
            "state": self.old_state[-1],
            "next_state": self.state[-1],
            "reward": reward,
            "step": self.global_steps,
        }
        self.history.append(self.transition)
        self._increase_global_step()
        return observation, self.state, reward

    def generate_objects(self):
        n_states = len(self.xy_combination)
        object_indices = np.random.randint(0, self.n_objects, size=n_states)
        objects = np.eye(self.n_objects)[object_indices]
        return objects

    def make_object_observation(self, pos):
        index = self.pos_to_state(np.array(pos))
        obj = self.objects[index]
        return [index, obj, pos]

    def pos_to_state(self, pos):
        """
        Convert an (x,y) position to a discretised state index by nearest neighbour among self.xy_combination.
        """
        if not self.use_behavioural_data and np.shape(pos) == (2, 2):
            pos = pos[0]
        elif self.use_behavioural_data and len(pos) > 2:
            pos = pos[:2]

        idx = np.argmin(np.sum((self.xy_combination - pos) ** 2, axis=1))
        return idx

    def validate_action(self, pre_state, action, new_state):
        crossed_wall = False
        for wall in self.wall_list:
            new_state, crossed = check_crossing_wall(
                pre_state=pre_state, new_state=np.asarray(new_state), wall=wall, wall_closenes=0.0
            )
            crossed_wall = crossed or crossed_wall

        nearest_idx = np.argmin(np.sum((self.xy_combination - new_state) ** 2, axis=1))
        new_state = self.xy_combination[nearest_idx]

        return new_state, crossed_wall

    def plot_trajectory(
        self,
        history_data: list = None,
        ax=None,
        return_figure: bool = False,
        save_path: str = None,
        plot_every: int = 1,
    ):
        if history_data is None:
            history_data = self.history

        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(8, 6))

        if len(history_data) != 0:
            state_history = [s["state"] for s in history_data]
            x = [s[0] for s in state_history]
            y = [s[1] for s in state_history]
            ax = make_plot_trajectories(self.arena_limits, np.asarray(x), np.asarray(y), ax, plot_every)

        for wall in self.default_walls:
            ax.plot(wall[:, 0], wall[:, 1], "C3", lw=3)
        for wall in self.custom_walls:
            ax.plot(wall[:, 0], wall[:, 1], "C0", lw=3)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")

        if return_figure:
            return ax, f
        else:
            return ax

    def render(self, history_length=30, display=True):
        f, ax = plt.subplots(1, 1, figsize=(8, 6))
        canvas = FigureCanvas(f)
        history = self.history[-history_length:]
        ax = self.plot_trajectory(history_data=history, ax=ax)
        canvas.draw()
        image = np.frombuffer(canvas.buffer_rgba(), dtype="uint8")
        image = image.reshape(f.canvas.get_width_height()[::-1] + (4,))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        print(image.shape)
        cv2.imshow("2D_env", image)
        cv2.waitKey(10)
