import os
import pickle
import shutil
import sys
import traceback

import pandas as pd

from neuralplayground.agents import AgentCore
from neuralplayground.arenas import Environment
from neuralplayground.config import STATE_LABELS
from neuralplayground.utils import check_dir, get_date_time


class SimulationManager(object):
    """Class to manage the runs of multiple combinations of agents, environments, parameters and training loops

    Attributes
    ----------
    simulation_list: list of SingleSim objects
        List of SingleSim objects to run
    runs_per_sim: int
        Number of runs per simulation
    manager_id: str
        ID of the simulation manager
    results_path: str
        Path to the results of the simulation manager
    verbose: bool
        If True, print information about the simulation manager

    Methods
    -------
    generate_sim_paths()
        Generate the paths for the simulations
    run_all()
        Run all the simulations in the simulation manager list, runs_per_sim times
    check_run_status()
        Prints the status of the simulations in the simulation manager
    """

    def __init__(
        self,
        simulation_list: list = [],
        runs_per_sim: int = 5,
        manager_id: str = "default_comparison",
        verbose: bool = False,
        existing_simulation: str = None,
    ):
        """Initialize the simulation manager

        Parameters
        ----------
        simulation_list: list of SingleSim objects
            List of SingleSim objects to run (defined below in the code)
        runs_per_sim: int
            Number of runs per simulation
        manager_id: str
            ID of the simulation manager
        verbose: bool
            If True, print information about the simulation manager
        existing_simulation: str
            Path to an existing simulation manager, it will load the parameters from the existing simulation if provided
        """
        if existing_simulation is not None:
            self._init_existing_sim(existing_simulation)
            return

        self.simulation_list = simulation_list
        self.runs_per_sim = runs_per_sim
        self.manager_id = manager_id
        self.results_path = manager_id
        self.verbose = verbose
        if self.verbose:
            print(self)

    def _init_existing_sim(self, existing_simulation: str):
        """Initialize the simulation manager with existing simulations"""
        param_sims = os.path.join(existing_simulation, "simulation.params")
        self.__dict__ = pickle.load(open(param_sims, "rb"))

    def generate_sim_paths(self):
        """Generate the paths for the simulations"""
        self.full_results_path = self.results_path
        self.simulation_paths = []
        self.run_paths = []
        str_path = self.full_results_path
        # creating the path for the simulation manager
        for sim in self.simulation_list:
            sim_path = os.path.join(self.full_results_path, sim.simulation_id)
            self.simulation_paths.append(sim_path)
            str_path += f"\n  {sim_path}"
            # creating the path for each run
            for run in range(self.runs_per_sim):
                # writing path for each run index and date time
                run_path = os.path.join(sim_path, f"run_{run}_{get_date_time()}")
                check_dir(run_path)
                self.run_paths.append(run_path)
                # writing state log
                state_log_path = os.path.join(run_path, "state.log")
                sim._update_log_state(message="in_queue", save_path=state_log_path)
                str_path += f"\n    {run_path}"

        self.save_params(self.full_results_path)

        if self.verbose:
            print(str_path)

    def __str__(self):
        """Print the simulation manager information"""
        sim_list = [sim.simulation_id for sim in self.simulation_list]
        mssg_str = (
            f'SimulationManager "{self.manager_id}" \nwith {sim_list} simulations'
            + f" \nand {self.runs_per_sim} runs per simulation"
        )
        return mssg_str

    def run_all(self):
        """Run all the SingleSim in the simulation manager list, runs_per_sim times"""
        # running all the simulations
        for sim_index, sim in enumerate(self.simulation_list):
            # running all the runs for each simulation
            for run_index in range(self.runs_per_sim):
                sim_path = self.run_paths[run_index + sim_index * self.runs_per_sim]
                print("Running simulation at path:")
                print(sim_path)
                self._logged_run(sim, sim_path)

    def save_params(self, save_path: str):
        """Save the parameters of the simulation manager"""
        save_path_params = os.path.join(save_path, "simulation.params")
        pickle.dump(self.__dict__, open(save_path_params, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    def check_run_status(self):
        """Check the status of all the simulations"""
        print("Checking status of simulations")
        str_path = self.full_results_path
        for i, sim in enumerate(self.simulation_list):
            sim_path = os.path.join(self.full_results_path, sim.simulation_id)
            self.simulation_paths.append(sim_path)
            str_path += f"\n  {sim_path}"
            for run in range(self.runs_per_sim):
                run_path = self.run_paths[run + i * self.runs_per_sim]
                state_log_path = os.path.join(run_path, "state.log")
                state_str = self._get_state(state_log_path)
                state_str_color = STATE_LABELS[state_str]
                str_path += f"\n    {run_path}: {state_str_color}"
        print(str_path)

    def show_logs(self, simulation_index: int = 0, log_type: str = "error"):
        """Show the logs of a simulation"""
        sim_path = self.simulation_paths[simulation_index]
        sim_object = self.simulation_list[simulation_index]
        print(f"Showing logs for simulation at path: {sim_path}")
        for run in range(self.runs_per_sim):
            run_path = self.run_paths[run + simulation_index * self.runs_per_sim]
            print("log for run: ", run)
            sim_object.show_logs(run_path, log_type=log_type)

    def rerun_simulation(self, simulation_index: int = 0):
        str_path = "rerun simulation at path: "
        sim_path = self.simulation_paths[simulation_index]
        sim_object = self.simulation_list[simulation_index]
        str_path += f"\n  {sim_path}"
        shutil.rmtree(sim_path)
        # creating the path for each run
        for run in range(self.runs_per_sim):
            # writing path for each run index and date time
            run_path = os.path.join(sim_path, f"run_{run}_{get_date_time()}")
            check_dir(run_path)
            self.run_paths[run + simulation_index * self.runs_per_sim] = run_path
            # writing state log
            state_log_path = os.path.join(run_path, "state.log")
            sim_object._update_log_state(message="in_queue", save_path=state_log_path)
            str_path += f"\n    {run_path}"
        print(str_path)

        for run in range(self.runs_per_sim):
            sim_path = self.run_paths[run + simulation_index * self.runs_per_sim]
            sim = sim_object
            print("Running simulation at path:")
            print(sim_path)
            self._logged_run(sim, sim_path)

    def _get_state(self, state_path):
        """Get the state of the simulation from the state log"""
        with open(state_path, "r") as f:
            state_str = f.readline().split("\n")[0]
        return state_str

    def _logged_run(self, sim, sim_path):
        # setting run logs and error logs
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            sim.run_sim(save_path=sim_path)
        except Exception:
            # Logging the error
            sim._update_log_state(message="error", save_path=os.path.join(sim_path, "state.log"))
            sys.stdout = open(os.path.join(sim_path, "error.log"), "a")
            print(traceback.format_exc())
            sys.stdout.close()
        # Recover the original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr


class SingleSim(object):
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
        training_loop=None,
        training_loop_params=None,
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
        self.training_loop = training_loop
        self.training_loop_params = training_loop_params
        self.simulation_id = simulation_id

    def run_sim(self, save_path: str = None):
        """Run the simulation and save the results in save_path

        Parameters
        ----------
        save_path : str
            The path where the results of the simulation will be saved
        """

        # Setting the save path and logs
        if save_path is None:
            save_path = os.path.join(os.getcwd(), "results_sim")
        check_dir(save_path)
        run_log_path = os.path.join(save_path, "run.log")
        error_log_path = os.path.join(save_path, "error.log")
        state_log_path = os.path.join(save_path, "state.log")

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        sys.stdout = open(run_log_path, "w")
        sys.stderr = open(error_log_path, "w")

        # Initializing the state log
        self._update_log_state("running", state_log_path)

        # Saving simulation parameters
        print("---> Saving simulation parameters")
        self.save_params(save_path)

        # Initializing models
        print("---> Initializing models")
        agent, env = self._init_models()

        # Training loop
        print("---> Training loop")
        trained_agent, trained_env, training_hist = self.training_loop(agent, env, **self.training_loop_params)

        # Saving models
        print("---> Saving models")
        self._save_models(save_path, trained_agent, trained_env, training_hist)

        print("---> Simulation finished")
        self._update_log_state("finished", state_log_path)

        # Closing logs
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    def _init_models(self):
        """Initialize the models"""
        agent = self.agent_class(**self.agent_params)
        env = self.env_class(**self.env_params)
        return agent, env

    def _save_models(self, save_path: str, agent: AgentCore, env: Environment, training_hist: dict):
        """Save the models and the training history"""
        agent.save_agent(os.path.join(save_path, "agent"))
        env.save_environment(os.path.join(save_path, "arena"))
        pickle.dump(training_hist, open(os.path.join(save_path, "training_hist.dict"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    def save_params(self, save_path: str):
        """Save the parameters of the simulation for reproducibility"""
        save_path_params = os.path.join(save_path, "params.dict")
        pickle.dump(self.__dict__, open(save_path_params, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    def load_params(self, load_path: str = None):
        """Load the parameters of the simulation for reproducibility"""
        if load_path is None:
            param_path = os.path.join(os.getcwd(), "results_sim", "params.dict")
        else:
            param_path = os.path.join(load_path, "params.dict")
        self.__dict__ = pd.read_pickle(param_path)

    def _update_log_state(self, message: str, save_path: str):
        """Update the state log of the simulation"""
        state_log = open(save_path, "w")
        state_log.write(message)
        state_log.close()

    def __str__(self):
        str_rep = f"Simulation: {self.simulation_id}\n"
        str_rep += f"Agent: {self.agent_class}\n"
        str_rep += f"Agent params: {self.agent_params}\n"
        str_rep += f"Environment: {self.env_class}\n"
        str_rep += f"Environment params: {self.env_params}\n"
        str_rep += f"Training loop: {self.training_loop}\n"
        str_rep += f"Training loop params: {self.training_loop_params}\n"
        return str_rep

    def load_results(self, results_path: str = None):
        """Load the results of a simulation from a path"""
        if results_path is None:
            results_path = os.path.join(os.getcwd(), "results_sim")
        self.load_params(os.path.join(results_path))
        trained_agent = pd.read_pickle(os.path.join(results_path, "agent"))
        trained_env = pd.read_pickle(os.path.join(results_path, "arena"))
        training_hist = pd.read_pickle(os.path.join(results_path, "training_hist.dict"))
        return trained_agent, trained_env, training_hist

    def show_logs(self, results_path: str = None, log_type: str = "error"):
        """Show the logs of the simulation"""
        if results_path is None:
            results_path = os.path.join(os.getcwd(), "results_sim")
        if log_type == "error":
            log_path = os.path.join(os.getcwd(), results_path, "error.log")
        elif log_type == "run":
            log_path = os.path.join(os.getcwd(), results_path, "run.log")
        elif log_type == "state":
            log_path = os.path.join(os.getcwd(), results_path, "state.log")
        else:
            raise ValueError("log_type must be either 'error' or 'run'")
        with open(log_path, "r") as log_file:
            print(log_file.read())
