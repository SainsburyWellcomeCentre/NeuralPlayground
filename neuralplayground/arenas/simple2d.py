import matplotlib as mpl
import matplotlib.pyplot as plt
from neuralplayground.arenas.arena_core import Environment
import numpy as np
from ..utils import check_crossing_wall


class Simple2D(Environment):
    """
    Methods
    ----------
    __init__(self, environment_name="2DEnv", **env_kwargs):
        Initialise the class
    reset(self):
        Reset the environment variables
    step(self, action):
        Increment the global step count of the agent in the environment and moves the agent in a random direction with a fixed step size
    plot_trajectory(self, history_data=None, ax=None):
        Plot the Trajectory of the agent in the environment

    Attribute
    ----------
    self.state: array
        Contains the x, y coordinate of the position and head direction of the agent (will be further developed)
        head_direction: ndarray
                Contains the x and y Coordinates of the position
        position: ndarray
                Contains the x and y Coordinates of the position
    self.history: dict
        Saved history over simulation steps (action, state, new_state, reward, global_steps)
    global_steps: int
        Counter of the number of steps in the environment
    room_width: int
        Size of the environment in the x direction
    room_depth: int
        Size of the environment in the y direction
    metadata: dict
        Dictionary containing the metadata of the children experiment
            doi: str
                Add the reference to the experiemental results
    observation: ndarray
        Fully observable environment, make_observation returns the state
        Array of the observation of the agent in the environment (Could be modified as the environments are evolves)
    action: ndarray (env_dim,env_dim)
        Array containing the action of the agent
        In this case the delta_x and detla_y increment to the respective coordinate x and y of the position
    reward: int
        The reward that the animal recieves in this state
    agent_step_size: float
         Size of the step when executing movement, agent_step_size*global_steps will give a measure of the total distance traversed by the agent

     """
    def __init__(self, environment_name="2DEnv", **env_kwargs):
        """ Initialise the class

        Parameters
        ----------
        env_kwargs: dict
        Dictionary with parameters of the experiment of the children class
            time_step_size:float
                Time_step_size*global_step_number will give a measure of the time in the experimental setting (s)
            agent_step_size: float
                agent_step_size*global_step_number will give a measure of the distance in the experimental setting
            room_width: int
                Size of the environment in the y direction
            room_depth: int
                Size of the environment in the y direction

        environment_name: str
            Name of the specific instantiation of the Simple2D class

        """
        super().__init__(environment_name, **env_kwargs)
        self.metadata = {"env_kwargs": env_kwargs}
        # self.room_width, self.room_depth = env_kwargs["room_width"], env_kwargs["room_depth"]
        self.arena_x_limits, self.arena_y_limits = env_kwargs["arena_x_limits"], env_kwargs["arena_y_limits"]
        self.arena_limits = np.array([[self.arena_x_limits[0], self.arena_x_limits[1]],
                                      [self.arena_y_limits[0], self.arena_y_limits[1]]])
        self.room_width = np.diff(self.arena_x_limits)[0]
        self.room_depth = np.diff(self.arena_y_limits)[0]
        self.agent_step_size = env_kwargs["agent_step_size"]
        self.state_dims_labels = ["x_pos", "y_pos"]
        self.reset()
        self._create_default_walls()
        self._create_custom_walls()
        self.wall_list = self.default_walls + self.custom_walls

    def _create_default_walls(self):
        self.default_walls = []
        self.default_walls.append(np.array([[self.arena_limits[0, 0], self.arena_limits[1, 0]],
                                           [self.arena_limits[0, 0], self.arena_limits[1, 1]]]))
        self.default_walls.append(np.array([[self.arena_limits[0, 1], self.arena_limits[1, 0]],
                                           [self.arena_limits[0, 1], self.arena_limits[1, 1]]]))
        self.default_walls.append(np.array([[self.arena_limits[0, 0], self.arena_limits[1, 0]],
                                           [self.arena_limits[0, 1], self.arena_limits[1, 0]]]))
        self.default_walls.append(np.array([[self.arena_limits[0, 0], self.arena_limits[1, 1]],
                                           [self.arena_limits[0, 1], self.arena_limits[1, 1]]]))

    def _create_custom_walls(self):
        self.custom_walls = []

    def reset(self, random_state=False, custom_state=None):
        """ Reset the environment variables

        Returns
        -------
        observation: ndarray
            Because this is a fully observable environment, make_observation returns the state of the environment
            Array of the observation of the agent in the environment (Could be modified as the environments are evolves)

        self.state: ndarray
            self.pos: ndarray (env_dim,)
                Vector of the x and y coordinate of the position of the animal in the environment
            self.head_dir: ndarray (env_dim,)
                Vector of the x and y coordinate of the animal head position in the environment
        """
        self.global_steps = 0
        self.history = []
        if random_state:
            self.state = [np.random.uniform(low=self.arena_limits[0, 0], high=self.arena_limits[0, 1]),
                          np.random.uniform(low=self.arena_limits[1, 0], high=self.arena_limits[1, 1])]
        else:
            self.state = [0, 0]
        self.state = np.array(self.state)

        if custom_state is not None:
            self.state = custom_state
        # Fully observable environment, make_observation returns the state
        observation = self.make_observation()
        return observation, self.state

    def step(self, action, normalize_step=False):
        """ Increment the global step count of the agent in the environment and updates the position of the agent according 
        to the recordings of the specific chosen session (Action is ignored in this case)

        Parameters
        ----------
        action: ndarray (env_dim,env_dim)
            Array containing the action of the agent
            In this case the delta_x and detla_y increment to the respective coordinate x and y of the position

        Returns
        -------
        reward: int
            The reward that the animal recieves in this state
        new_state: tuple
            Update the state with the updated vector of coordinate x and y of position and head directions espectively
        observation: ndarray
            Fully observable environment, make_observation returns the state
            Array of the observation of the agent in the environment ( Could be modified as the environments are evolves)

        """
        self.global_steps += 1
        if normalize_step:
            action = action/np.linalg.norm(action)
            new_state = self.state + self.agent_step_size*action
        else:
            new_state = self.state + action
        new_state, valid_action = self.validate_action(self.state, action, new_state)
        reward = 0  # If you get reward, it should be coded here
        transition = {"action": action, "state": self.state, "next_state": new_state,
                      "reward": reward, "step": self.global_steps}
        self.history.append(transition)
        self.state = new_state
        observation = self.make_observation()
        return observation, new_state, reward

    def validate_action(self, pre_state, action, new_state):
        """

        Parameters
        ----------
        pre_state : (2,) 2d-ndarray
            2d position of pre-movement
        new_state : (2,) 2d-ndarray
            2d position of post-movement

        Returns
        -------
        new_state: (2,) 2d-ndarray
            corrected new state. If it is not crossing the wall, then the new_state stays the same, if the state cross the
            wall, new_state will be corrected to a valid place without crossing the wall
        valid_action: bool
            True if the change in state cross a wall
        """
        valid_action = True
        for wall in self.wall_list:
            new_state, new_valid_action = check_crossing_wall(pre_state=pre_state, new_state=new_state, wall=wall)
            valid_action = new_valid_action and valid_action
        return new_state, valid_action

    def plot_trajectory(self, history_data=None, ax=None, return_figure=False):
        """ Plot the Trajectory of the agent in the environment

        Parameters
        ----------
        history_data: None
            default to access to the saved history of positions in the environment
        ax: None
            default to create ax

        Returns
        -------
        Returns a plot of the trajectory of the animal in the environment
        """
        if history_data is None:
            history_data = self.history
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(8, 6))

        for wall in self.default_walls:
            ax.plot(wall[:, 0], wall[:, 1], "C3", lw=3)

        for wall in self.custom_walls:
            ax.plot(wall[:, 0], wall[:, 1], "C0", lw=3)

        if len(history_data) != 0:
            state_history = [s["state"] for s in history_data]
            next_state_history = [s["next_state"] for s in history_data]
            starting_point = state_history[0]
            ending_point = next_state_history[-1]

            cmap = mpl.cm.get_cmap("plasma")
            norm = plt.Normalize(0, len(state_history))

            aux_x = []
            aux_y = []
            for i, s in enumerate(state_history):
                x_ = [s[0], next_state_history[i][0]]
                y_ = [s[1], next_state_history[i][1]]
                aux_x.append(s[0])
                aux_y.append(s[1])
                ax.plot(x_, y_, "-", color=cmap(norm(i)), alpha=0.6)

            # ax.set_xticks([])
            # ax.set_yticks([])
            sc = ax.scatter(aux_x, aux_y, c=np.arange(len(aux_x)), vmin=0, vmax=len(aux_x), cmap="plasma", alpha=0.6, s=0.1)
            cbar = plt.colorbar(sc, ax=ax,ticks = [0, len(state_history)])
            cbar.ax.set_ylabel('N steps', rotation=270,fontsize=16)
            cbar.ax.set_yticklabels([0,len(state_history)],fontsize=16)
        if return_figure:
            return f, ax
        else:
            return ax
