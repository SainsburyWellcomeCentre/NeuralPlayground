import sys
sys.path.append("../")
import matplotlib as mpl
import matplotlib.pyplot as plt
from environments.core import Environment
import numpy as np
from environments.experiments.behavioral_data import SargoliniData, FullSargoliniData,FullHaftingData


class Simple2D(Environment):
    """
    Methods
    ----------
    __init__(self, environment_name="2DEnv", **env_kwargs):
        Initialise the class
    reset(self):
        Reset the environment variables
    step(self, action):
        Increment the global step count of the agent in the environment
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
        agent_step_size*global_steps will give a measure of the distance in the experimental setting



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
        self.room_width, self.room_depth = env_kwargs["room_width"], env_kwargs["room_depth"]
        self.arena_limits = np.array([[-self.room_width/2, self.room_width/2],
                                      [-self.room_depth/2, self.room_depth/2]])
        self.agent_step_size = env_kwargs["agent_step_size"]
        self.state_dims_labels = ["x_pos", "y_pos"]
        self.reset()

    def reset(self):
        """ Reset the environment variables

        Returns
        -------
        observation: ndarray
            Fully observable environment, make_observation returns the state
            Array of the observation of the agent in the environment (Could be modified as the environments are evolves)

        self.state: ndarray
            self.pos: ndarray (env_dim,)
                Vector of the x and y coordinate of the position of the animal in the environment
            self.head_dir: ndarray (env_dim,)
                Vector of the x and y coordinate of the animal head position in the environment
        """
        self.global_steps = 0
        self.history = []
        self.state = [0,0]
        self.state = np.array(self.state)
        # Fully observable environment, make_observation returns the state
        observation = self.make_observation()
        return observation, self.state

    def step(self, action):
        """ Increment the global step count of the agent in the environment (Action is ignored in this case)

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
        new_state = np.array([np.clip(new_state[0], a_min=-self.room_width/2, a_max=self.room_width/2),
                              np.clip(new_state[1], a_min=-self.room_depth/2, a_max=self.room_depth/2)])
        reward = 0  # If you get reward, it should be coded here
        transition = {"action": action, "state": self.state, "next_state": new_state,
                      "reward": reward, "step": self.global_steps}
        self.history.append(transition)
        self.state = new_state
        observation = self.make_observation()
        return observation, new_state, reward


    def plot_trajectory(self, history_data=None, ax=None):
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

        ax.plot([-self.room_width/2, self.room_width/2],
                [-self.room_depth/2, -self.room_depth/2], "r", lw=2)
        ax.plot([-self.room_width/2, self.room_width/2],
                [self.room_depth/2, self.room_depth/2], "r", lw=2)
        ax.plot([-self.room_width/2, -self.room_width/2],
                [-self.room_depth/2, self.room_depth/2], "r", lw=2)
        ax.plot([self.room_width / 2, self.room_width / 2],
                [-self.room_depth / 2, self.room_depth / 2], "r", lw=2)

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
        Increment the global step count of the agent in the environment

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
        limits = np.abs(np.diff(self.arena_limits, axis=1))
        self.room_width = limits[0, 0]
        self.room_depth = limits[1, 0]
        env_kwargs["room_width"] = self.room_width
        env_kwargs["room_depth"] = self.room_depth
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
        """ Increment the global step count of the agent in the environment (Action is ignored in this case)

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
        Increment the global step count of the agent in the environment

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
        session: int
            Session number to run from the experimental data
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
        self.room_width, self.room_depth = np.abs(np.diff(self.arena_limits, axis=1))
        self.room_width = self.room_width[0]
        self.room_depth = self.room_depth[0]
        env_kwargs["room_width"] = self.room_width
        env_kwargs["room_depth"] = self.room_depth
        env_kwargs["agent_step_size"] = 1/50  # In seconds
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

        self.global_steps = 0
        self.global_time = 0
        self.history = []
        self.pos, self.head_dir = self.data.position[0, :], self.data.head_direction[0, :]
        self.state = np.concatenate([self.pos, self.head_dir])
        # Fully observable environment, make_observation returns the state
        observation = self.make_observation()
        return observation, self.state

    def step(self, action):
        """ Increment the global step count of the agent in the environment (Action is ignored in this case)

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


class Hafting2008(Simple2D):
    """
    Methods
    ----------
    __init__
       Initialise the class
    reset(self):
      Reset the environment variables
    step(self, action):
       Increment the global step count of the agent in the environment

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
        self.room_width, self.room_depth = np.abs(np.diff(self.arena_limits, axis=1))
        env_kwargs["room_width"] = self.room_width
        env_kwargs["room_depth"] = self.room_depth
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
        """ Increment the global step count of the agent in the environment (Action is ignored in this case)

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

if __name__ == "__main__":
    from models.weber_and_sprekeler import ExcInhPlasticity
    from environments.environments.simple2d import Simple2D, Sargolini2006, Hafting2008, BasicSargolini2006
    import sys
    sys.path.append("../")
    from tqdm import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    data_path = "../experiments/sargolini2006/"

    env = Sargolini2006(data_path=data_path,
                             time_step_size=0.1,
                             agent_step_size=None)


    exc_eta = 2e-4
    inh_eta = 8e-4
    model_name = "model_example"
    sigma_exc = np.array([0.05, 0.05])
    sigma_inh = np.array([0.1, 0.1])
    Ne = 4900
    Ni = 1225
    Nef = 1
    Nif = 1
    alpha_i = 1
    alpha_e = 1
    we_init = 1.0
    wi_init = 1.5
    agent_step_size = 0.1
    agentsimple = ExcInhPlasticity(model_name=model_name, exc_eta=exc_eta, inh_eta=inh_eta, sigma_exc=sigma_exc,
                                   sigma_inh=sigma_inh, Ne=Ne, Ni=Ni, agent_step_size=agent_step_size, ro=1,
                                   Nef=Nef, Nif=Nif, room_width=env.room_width, room_depth=env.room_depth,
                                   alpha_i=alpha_i, alpha_e=alpha_e, we_init=we_init, wi_init=wi_init)

    plot_every = 2
    total_iters = 10

    n_steps = 50000
    print(agentsimple.room_width)
    obs, state = env.reset()
    for i in tqdm(range(n_steps)):
        # Observe to choose an action
        obs = obs[:2]
        action = agentsimple.act(obs)
        # rate = agent.update()
        agentsimple.update()
        # Run environment for given action
        obs, state, reward = env.step(action)
        total_iters += 1

        if i % plot_every == 0:
            ax = env.plot_trajectory()
            agentsimple.plot_rates()
            agentsimple.plot_rates(save_path="../model/figures/pre_processed_iter_" + str(i) + ".pdf")

