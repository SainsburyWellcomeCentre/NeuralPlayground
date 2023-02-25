import copy
from neuralplayground.arenas import Simple2D
import numpy as np
from neuralplayground.experiments import Hafting2008Data


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
    def __init__(self, use_behavioral_data: bool = False, data_path: str = None, recording_index: int = None,
                 environment_name: str = "Hafting2008", verbose: bool = False, **env_kwargs):
        """ Initialise the class

        Parameters
        ----------
        use_behavioral_data: bool
            If True, then uses the animal trajectories recorded in Hafting 2008
        data_path: str
            if None, load the data of corresponding experiment available in the package,
            else load data from given path
        recording_index: int
            if None, load data from default recording index of corresponding experiment class
        environment_name: str
            Name of the specific instantiation of the Hafting2008 class
        verbose:
            Set to True to show the information of the class
        env_kwargs: dict
            Leave empty in this class, the arena parameters and sampling values are set to resemble the experiment
            For full control over these parameters use Simple2D class
        """

        self.data_path = data_path
        self.environment_name = environment_name
        self.use_behavioral_data = use_behavioral_data
        self.experiment = Hafting2008Data(data_path=self.data_path, experiment_name=self.environment_name,
                                          verbose=verbose, recording_index=recording_index)
        self.arena_limits = self.experiment.arena_limits
        self.arena_x_limits, self.arena_y_limits = self.arena_limits[0, :], self.arena_limits[1, :]
        env_kwargs["arena_x_limits"] = self.arena_x_limits
        env_kwargs["arena_y_limits"] = self.arena_y_limits
        env_kwargs["agent_step_size"] = 1.0
        env_kwargs["time_step_size"] = 1/50  # Taken from experiment, 50 Hz movement sampling
        super().__init__(environment_name, **env_kwargs)
        if self.use_behavioral_data:
            self.state_dims_labels = ["x_pos", "y_pos", "head_direction_x", "head_direction_y"]

    def reset(self, random_state: bool = False, custom_state: np.ndarray = None):
        """ Reset the environment variables. If using behavioral data, it will reset the position to the
        initial position of the trajectory recorded in the experiment

        Parameters
        ----------
        random_state: bool
            If True, sample a new position uniformly within the arena, use default otherwise
        custom_state: np.ndarray
            If given, use this array to set the initial state

        Returns
        ----------
        observation: ndarray
            Because this is a fully observable environment, make_observation returns the state of the environment
            Array of the observation of the agent in the environment (Could be modified as the environments are evolves)

        self.state: ndarray (2,)
            Vector of the x and y coordinate of the position of the animal in the environment
        """
        # Reset to first position recorded in this session
        if self.use_behavioral_data:
            self.pos, self.head_dir = self.experiment.position[0, :], self.experiment.head_direction[0, :]
            custom_state = np.concatenate([self.pos, self.head_dir])
            return super().reset(random_state=False, custom_state=custom_state)
        # Default reset
        else:
            return super().reset(random_state=random_state, custom_state=custom_state)

    def set_animal_data(self, recording_index: int = 0, tolerance: float = 1e-10, keep_history: bool = True):
        """ Set position and head direction to be used by the Arena Class,
        See neuralplayground.experiments classes """

        self.experiment.set_animal_data(recording_index=recording_index, tolerance=tolerance)
        if keep_history:
            prev_hist = copy.copy(self.history)
            self.reset()
            self.history = prev_hist
        else:
            self.reset()

    def show_data(self, full_dataframe: bool = False):
        """ Print of available data recorded in the experiment

        Parameters
        ----------
        full_dataframe: bool
            if True, it will show all available experiment, a small sample otherwise

        Returns
        -------
        recording_list: Pandas dataframe
            List of available experiment, columns with rat_id, recording session and recorded variables
        """
        self.experiment.show_data(full_dataframe=full_dataframe)

    def step(self, action: np.ndarray, normalize_step: bool = False, skip_every: int = 10):
        """ Runs the environment dynamics. Increasing global counters.
        Given some action, return observation, new state and reward.
        If using behavioral data, the action argument is ignored, and instead inferred from the recorded trajectory
        in the experiment.

        Parameters
        ----------
        action: ndarray (2,)
            Array containing the action of the agent, in this case the delta_x and detla_y increment to position
        normalize_step: bool
            If true, the action is normalized to have unit size, then scaled by the agent step size
        skip_every: int
            When using behavioral data, the next state will be the position and head direction
            "skip_every" recorded steps after the current one, essentially reducing the sampling rate

        Returns
        -------
        reward: float
            The reward that the animal receives in this state
        new_state: ndarray
            Update the state with the updated vector of coordinate x and y of position and head directions respectively
        observation: ndarray
            Array of the observation of the agent in the environment
        """
        if not self.use_behavioral_data:
            return super().step(action)

        # In this case, the action is ignored and computed from the step in behavioral data recorded from the experiment
        if self.global_steps*skip_every >= self.experiment.position.shape[0]-1:
            self.global_steps = np.random.choice(np.arange(skip_every))
        # Time elapsed since last reset
        self.global_time = self.global_steps*self.time_step_size

        # New state as "skip every" steps after the current one in the recording
        new_state = self.experiment.position[self.global_steps * (skip_every), :], \
            self.experiment.head_direction[self.global_steps * (skip_every), :]
        new_state = np.concatenate(new_state)

        # Inferring action from recording
        action = new_state - self.state
        reward = self.reward_function(action, state=self.state)
        transition = {"action": action, "state": self.state, "next_state": new_state,
                      "reward": reward, "step": self.global_steps}
        self.history.append(transition)
        self.state = new_state
        observation = self.make_observation()
        self._increase_global_step()
        return observation, new_state, reward
