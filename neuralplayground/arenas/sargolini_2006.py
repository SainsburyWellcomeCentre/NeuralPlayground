import copy

from neuralplayground.arenas import Hafting2008
from neuralplayground.experiments import (
    Sargolini2006Data,
    SargoliniDataTrajectory,
)


class Sargolini2006(Hafting2008):
    """Arena resembling Sargolini2006 experimental setting

    Methods
    ----------
    __init__(self, use_behavioral_data: bool = False, data_path: str = None, recording_index: int = None,
                 environment_name: str = "Sargolini2006", verbose: bool = False, experiment_class=Sargolini2006Data,
                 **env_kwargs):
         Initialise the class
    reset(self):
        Reset the environment variables
    step(self, action):
        Increment the global step count of the agent in the environment and updates the position of the agent according
        to the recordings of the specific chosen session

    Attributes
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

    def __init__(
        self,
        use_behavioral_data: bool = False,
        data_path: str = None,
        recording_index: int = None,
        environment_name: str = "Sargolini2006",
        verbose: bool = False,
        experiment_class=Sargolini2006Data,
        **env_kwargs,
    ):
        super().__init__(
            use_behavioral_data,
            data_path,
            recording_index,
            environment_name,
            verbose,
            experiment_class,
            **env_kwargs,
        )


class BasicSargolini2006(Hafting2008):
    def __init__(
        self,
        use_behavioral_data: bool = False,
        data_path: str = None,
        recording_index: int = None,
        environment_name: str = "Sargolini2006",
        verbose: bool = False,
        experiment_class=SargoliniDataTrajectory,
        **env_kwargs,
    ):
        super().__init__(
            use_behavioral_data,
            data_path,
            recording_index,
            environment_name,
            verbose,
            experiment_class,
            **env_kwargs,
        )

    def set_animal_data(
        self,
        recording_index: int = 0,
        tolerance: float = 1e-10,
        keep_history: bool = True,
    ):
        """No recording index to use in this particular dataset"""
        if keep_history:
            prev_hist = copy.copy(self.history)
            self.reset()
            self.history = prev_hist
        else:
            self.reset()

    def show_data(self, full_dataframe: bool = False):
        print("no dataframe with sessions, just pre-processed positions of all trajectories")

    def plot_recording_tetr(self, **kwargs):
        print("No tetrode data available")
