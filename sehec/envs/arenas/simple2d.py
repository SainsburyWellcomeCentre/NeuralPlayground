import matplotlib as mpl
import matplotlib.pyplot as plt
from ..envcore import Environment
import numpy as np
from ..experiments.behavioral_data import SargoliniData, FullSargoliniData, FullHaftingData
from ...utils import check_crossing_wall


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

    def step(self, action):
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
        action = action/np.linalg.norm(action)
        new_state = self.state + self.agent_step_size*action
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

        state_history = [s["state"] for s in history_data]
        next_state_history = [s["next_state"] for s in history_data]
        starting_point = state_history[0]
        ending_point = next_state_history[-1]
        print(starting_point)

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

        sc = ax.scatter(aux_x, aux_y, c=np.arange(len(state_history)),
                        vmin=0, vmax=len(state_history), cmap="plasma", alpha=0.6)

        cbar = plt.colorbar(sc, ax=ax)
        cbar.ax.set_ylabel('N steps', rotation=270)
        if return_figure:
            return f, ax
        else:
            return ax


class BasicSargolini2006(Simple2D):
    """
    Methods
    ----------
    __init__(self, data_path="Sargolini2006/", environment_name="Sargolini2006", **env_kwargs):
        Initialise the class
    reset(self):
        Reset the environment variables
    step(self, action):
        Increment the global step count of the agent in the environment and updates the position of the agent

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
              """

    def __init__(self, data_path="Sargolini2006/", environment_name="Sargolini2006", **env_kwargs):
        """ Initialise the class

        Parameters
        ----------
        env_kwargs: dict
            Dictionary with parameters of the experiment Sargolini 2006 https://doi.org/10.1126/science.1125572
            time_step_size:float
                Time_step_size*global_step_number will give a measure of the time in the experimental setting (s)
            agent_step_size: float
                agent_step_size*global_step_number will give a measure of the distance in the experimental setting
        data_path: str
            Path to the environment data
        environment_name: str
            Name of the specific instantiation of the BasicSargolini2006
        """

        self.data_path = data_path
        self.environment_name = environment_name
        self.data = SargoliniData(data_path=self.data_path, experiment_name=self.environment_name)
        self.arena_limits = self.data.arena_limits
        self.arena_x_limits, self.arena_y_limits = self.arena_limits[0, :], self.arena_limits[1, :]
        env_kwargs["arena_x_limits"] = self.arena_x_limits
        env_kwargs["arena_y_limits"] = self.arena_y_limits
        env_kwargs["agent_step_size"] = 1/50  # In seconds
        super().__init__(environment_name, **env_kwargs)
        self.metadata["doi"] = "https://doi.org/10.1126/science.1125572"
        self.total_number_of_steps = self.data.position.shape[0]
        self.state_dims_labels = ["x_pos", "y_pos", "head_direction_x", "head_direction_y"]

    def reset(self):
        """ Reset the environment variables

        Returns
        -------
        observation: ndarray
            Fully observable environment, make_observation returns the state
            Array of the observation of the agent in the environment ( Could be modified as the environments are evolves)
        self.state: ndarray
            self.pos: ndarray (env_dim,)
                Vector of the x and y coordinate of the position of the animal in the environment
            self.head_dir: ndarray (env_dim,)
                Vector of the x and y coordinate of the animal head position in the environment
        """
        self.global_steps = 0
        self.global_time = 0
        self.history = []
        self.pos, self.head_dir = self.data.position[0, :], self.data.head_direction[0, :]
        self.state = np.concatenate([self.pos, self.head_dir])
        # Fully observable environment, make_observation returns the state
        observation = self.make_observation()
        return observation, self.state

    def step(self, action):
        """ Increment the global step count of the agent in the environmentand updates the state of the agent to be the next
        recorded position (form the experiment) of the animal (Action is ignored in this case)

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
        if self.global_steps >= self.data.position.shape[0] - 1:
            self.global_steps = 0
        self.global_time = self.global_steps*self.agent_step_size
        reward = 0  # If you get reward, it should be coded here
        new_state = self.data.position[self.global_steps, :], self.data.head_direction[self.global_steps, :]
        new_state = np.concatenate(new_state)
        transition = {"action": action, "state": self.state, "next_state": new_state,
                      "reward": reward, "step": self.global_steps}
        self.history.append(transition)
        self.state = new_state
        observation = self.make_observation()
        self.global_steps += 1
        return observation, new_state, reward


class Sargolini2006(Simple2D):
    """
    Methods
    ----------
    __init__(self, data_path="sargolini2006/", environment_name="Sargolini2006", session=None, verbose=False, **env_kwargs):
         Initialise the class
    reset(self, sess=None):
        Reset the environment variables
    step(self, action):
        Increment the global step count of the agent in the environment and updates the position of the agent according 
        to the recordings of the specific chosen session

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

        """
    def __init__(self, data_path="sargolini2006/", environment_name="Sargolini2006", session=None, verbose=False, **env_kwargs):
        """  Initialise the class

        Parameters
        ----------
        data_path: str
            Path to the environment data
        environment_name:str
             Name of the specific instantiation of the Sargolini2006 class
        session: dict
            Dictionary with a rat number and session id to run from the experimental data
        verbose:bool
            Set to True to show the information of the class
        env_kwargs:dict
            Dictionary with parameters of the experiment Sargolini 2006 https://doi.org/10.1126/science.1125572
        time_step_size:float
                Time_step_size*global_step_number will give a measure of the time in the experimental setting (s)
        agent_step_size: float
                agent_step_size*global_step_number will give a measure of the distance in the experimental setting
        """
        self.data_path = data_path
        self.verbose= verbose
        self.environment_name = environment_name
        self.session = session
        self.data = FullSargoliniData(data_path=self.data_path, experiment_name=self.environment_name, verbose=verbose, session=session)
        self.arena_limits = self.data.arena_limits
        self.arena_x_limits, self.arena_y_limits = self.arena_limits[0, :], self.arena_limits[1, :]
        env_kwargs["arena_x_limits"] = self.arena_x_limits
        env_kwargs["arena_y_limits"] = self.arena_y_limits
        env_kwargs["agent_step_size"] = 1/50  # In seconds
        self.random_steps = env_kwargs["random_steps"]
        super().__init__(environment_name, **env_kwargs)
        self.metadata["doi"] = "https://doi.org/10.1126/science.1125572"
        self.total_number_of_steps = self.data.position.shape[0]
        self.state_dims_labels = ["x_pos", "y_pos", "head_direction_x", "head_direction_y"]

    def reset(self, sess=None):
        """  Reset the environment variables

        Parameters
        ----------
        sess: int
            Session number to run from the experimental data

        Returns
        -------
        observation: ndarray
            Fully observable environment, make_observation returns the state
            Array of the observation of the agent in the environment ( Could be modified as the environments are evolves)
        self.state: ndarray
            self.pos: ndarray (env_dim,)
                Vector of the x and y coordinate of the position of the animal in the environment
            self.head_dir: ndarray (env_dim,)
                Vector of the x and y coordinate of the animal head position in the environment
        """
        if not sess is None:
            self.data = FullSargoliniData(data_path=self.data_path, experiment_name=self.environment_name,
                                          verbose=self.verbose, session=sess)
        if self.random_steps:
            return super().reset()

        self.global_steps = 0
        self.global_time = 0
        self.history = []
        self.pos, self.head_dir = self.data.position[0, :], self.data.head_direction[0, :]
        self.state = np.concatenate([self.pos, self.head_dir])
        # Fully observable environment, make_observation returns the state
        observation = self.make_observation()
        return observation, self.state

    def step(self, action, skip_every=10):
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
        if self.random_steps:
            return super().step(action)
        if self.global_steps*skip_every >= self.data.position.shape[0]-1:
            self.global_steps = np.random.choice(np.arange(skip_every))
        self.global_time = self.global_steps*self.agent_step_size
        reward = 0  # If you get reward, it should be coded here
        new_state = self.data.position[self.global_steps*(skip_every), :], \
                    self.data.head_direction[self.global_steps*(skip_every), :]
        new_state = np.concatenate(new_state)
        transition = {"action": action, "state": self.state, "next_state": new_state,
                      "reward": reward, "step": self.global_steps}
        self.history.append(transition)
        self.state = new_state
        observation = self.make_observation()
        self.global_steps += 1
        return observation, new_state, reward


class Hafting2008(Simple2D):
    """
    Methods
    ----------
    __init__
       Initialise the class
    reset(self):
      Reset the environment variables
    step(self, action):
       Increment the global step count of the agent in the environment and updates the position of the agent according
       to the recordings of the specific chosen session

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

    """
    def __init__(self, data_path="Hafting2008/C43035A4-5CC5-44F2-B207-126922523FD9_1/", environment_name="Hafting2008", session=None, verbose=False, **env_kwargs):
        """ Initialise the class

        Parameters
        ----------
        data_path: str
            Path to the environment data
        environment_name: str
             Name of the specific instantiation of the Hafting2008 class
        session: int
            Session number to run from the experimental data
        verbose:
            Set to True to show the information of the class
        env_kwargs: dict
            Dictionary with parameters of the experiment Hafting 2008
        time_step_size:float
            Time_step_size*global_step_number will give a measure of the time in the experimental setting (s)
        agent_step_size: float
            agent_step_size*global_step_number will give a measure of the distance in the experimental setting

        """

        self.data_path = data_path
        self.environment_name = environment_name
        self.session = session
        self.data = FullHaftingData(data_path=self.data_path, experiment_name=self.environment_name, verbose=verbose)
        self.arena_limits = self.data.arena_limits
        self.arena_x_limits, self.arena_y_limits = self.arena_limits[0, :], self.arena_limits[1, :]
        env_kwargs["arena_x_limits"] = self.arena_x_limits
        env_kwargs["arena_y_limits"] = self.arena_y_limits
        env_kwargs["agent_step_size"] = 1/50  # In seconds
        super().__init__(environment_name, **env_kwargs)
        self.metadata["doi"] = "https://doi.org/10.1038/nature06957"
        self.state_dims_labels = ["x_pos", "y_pos", "head_direction_x", "head_direction_y"]

    def reset(self):
        """ Reset the environment variables

        Returns
        -------
        observation: ndarray
            Fully observable environment, make_observation returns the state
            Array of the observation of the agent in the environment ( Could be modified as the environments are evolves)
        self.state: ndarray
            self.pos: ndarray (env_dim,)
                Vector of the x and y coordinate of the position of the animal in the environment
            self.head_dir: ndarray (env_dim,)
                Vector of the x and y coordinate of the animal head position in the environment
        """
        self.global_steps = 0
        self.global_time = 0
        self.history = []
        self.pos, self.head_dir = self.data.position[0, :], self.data.head_direction[0, :]
        self.state = np.concatenate([self.pos, self.head_dir])
        # Fully observable environment, make_observation returns the state
        observation = self.make_observation()
        return observation, self.state

    def step(self, action):
        """ Increment the global step count of the agent in the environment
        and updates the position of the agent according to the recordings of the specific chosen session 
        (Action is ignored in this case)

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
        if self.global_steps >= self.data.position.shape[0]-1:
            self.global_steps = 0
        self.global_time = self.global_steps*self.agent_step_size
        reward = 0  # If you get reward, it should be coded here
        new_state = self.data.position[self.global_steps, :], self.data.head_direction[self.global_steps, :]
        new_state = np.concatenate(new_state)
        transition = {"action": action, "state": self.state, "next_state": new_state,
                      "reward": reward, "step": self.global_steps}
        self.history.append(transition)
        self.state = new_state
        observation = self.make_observation()
        self.global_steps += 1
        return observation, new_state, reward
