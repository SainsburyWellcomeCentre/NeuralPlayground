from sehec.arenas import Simple2D
import numpy as np
from sehec.experiments import FullHaftingData


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
