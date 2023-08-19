import os
import pickle

import pandas as pd


class PlotSim(object):
    """Single simulation object

    Attributes
    ----------
    agent_class : Agent
        The agent class to be used in the simulation
    agent_params : dict
        The parameters of the agent
    env_class : Environment
        The environment class to be used in the simulation
    env_params : dict
        The parameters of the environment
    training_loop : python function
        The training loop to be used in the simulation
    training_loop_params : dict
        The parameters of the training loop
    simulation_id : str
        The id of the simulation

    Methods
    -------
    run_sim(save_path: str)
        Run the simulation and save the results in save_path
    _init_model(save_path: str)
        Initialize the model and save it in save_path
    _save_model(save_path: str)
        Save the model in save_path
    save_params(save_path: str)
        Save the parameters of the simulation
    _update_log_state(message: str, save_path: str)
        Update the state log of the simulation
    """

    def __init__(
        self,
        agent_class=None,
        agent_params=None,
        env_class=None,
        env_params=None,
        plotting_loop_params=None,
        simulation_id: str = None,
    ):
        """Initialize the SingleSim object

        Parameters
        ----------
        agent_class : Agent
            The agent class to be used in the simulation
        agent_params : dict
            The parameters of the agent
        env_class : Environment
            The environment class to be used in the simulation
        env_params : dict
            The parameters of the environment
        training_loop : python function
            The training loop function to be used in the simulation
            see training_loops.py for examples in this same module
        training_loop_params : dict
            The parameters of the training loop (neither the agent nor the environment)
        simulation_id : str
            The id of the simulation
        """
        self.agent_class = agent_class
        self.agent_params = agent_params
        self.env_class = env_class
        self.env_params = env_params
        self.plotting_loop_params = plotting_loop_params
        self.simulation_id = simulation_id

    def plot_sim(self, save_path: str = None, n_walks: int = 1, random_state: bool = True, custom_state: list = None):
        """Run the simulation and save the results in save_path

        Parameters
        ----------
        save_path : str
            The path where the results of the simulation will be saved
        """
        # Initializing models
        print("---> Initializing models")
        agent, env = self._init_models()

        # Training loop
        print("---> Plotting loop")
        trained_agent, trained_env = self.tem_plotting_loop(agent, env, n_walks, random_state, custom_state)

        print("---> Finished")
        model_input, history, environments = trained_agent.collect_final_trajectory()
        environments = [trained_env.collect_environment_info(model_input, history, environments)]

        # Save environments and model_input using pickle
        with open(os.path.join(save_path, "NPG_environments.pkl"), "wb") as f:
            pickle.dump(environments, f)
        with open(os.path.join(save_path, "NPG_model_input.pkl"), "wb") as f:
            pickle.dump(model_input, f)
        return trained_agent, trained_env

    def _init_models(self):
        """Initialize the models"""
        agent = self.agent_class(**self.agent_params)
        env = self.env_class(**self.env_params)
        return agent, env

    def load_params(self, load_path: str = None):
        """Load the parameters of the simulation for reproducibility"""
        if load_path is None:
            param_path = os.path.join(os.getcwd(), "results_sim", "params.dict")
        else:
            param_path = os.path.join(load_path, "params.dict")
        self.__dict__ = pd.read_pickle(param_path)

    def tem_plotting_loop(self, agent, env, n_walks: int = 1000, random_state: bool = True, custom_state: list = None):
        # Run around environment
        observation, state = env.reset(random_state=random_state, custom_state=custom_state)
        while agent.n_walk < n_walks:
            if agent.n_walk % 1000 == 0 and agent.n_walk > 0:
                print(agent.n_walk)
            action = agent.batch_act(observation)
            observation, state, reward = env.step(action, normalize_step=True)
        return agent, env
