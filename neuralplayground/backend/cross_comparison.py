import os
import pickle
import sys

import pandas as pd

from neuralplayground.agents import AgentCore
from neuralplayground.arenas import Environment
from neuralplayground.utils import check_dir, get_date_time


class SimulationManager(object):
    """Class to manage the runs of multiple combinations of agents, environments, parameters and training loops"""

    def __init__(self, simulation_list, runs_per_sim: int, manager_id: str = "default_comparison", verbose: bool = False):
        self.simulation_list = simulation_list
        self.runs_per_sim = runs_per_sim
        self.manager_id = manager_id
        self.results_path = manager_id
        self.verbose = verbose
        if self.verbose:
            print(self)

    def generate_sim_paths(self):
        """Generate the paths for the simulations
        If the path does not exist, it will be created
        if the path exists, it will save the results in the existing path
        """
        self.full_results_path = self.results_path
        self.simulation_paths = []
        str_path = self.full_results_path
        for sim in self.simulation_list:
            sim_path = os.path.join(self.full_results_path, sim.simulation_id)
            self.simulation_paths.append(sim_path)
            str_path += f"\n  {sim_path}"
            for run in range(self.runs_per_sim):
                # writing path for each run index and date time
                run_path = os.path.join(sim_path, f"run_{run}_{get_date_time()}")
                check_dir(run_path)
                state_log_path = os.path.join(run_path, "state.log")
                sim._update_log_state(message="in_queue", save_path=state_log_path)
                str_path += f"\n    {run_path}"

        if self.verbose:
            print(str_path)

    def __str__(self):
        sim_list = [sim.simulation_id for sim in self.simulation_list]
        mssg_str = (
            f'SimulationManager "{self.manager_id}" \nwith {sim_list} simulations'
            + f" \nand {self.runs_per_sim} runs per simulation"
        )
        return mssg_str

    def run_all(self):
        pass

    def run_single_sim(self, sim_index):
        pass

    def check_runs(self):
        pass

    def set_config(self):
        pass


class SingleSim(object):
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
        self.agent_class = agent_class
        self.agent_params = agent_params
        self.env_class = env_class
        self.env_params = env_params
        self.training_loop = training_loop
        self.training_loop_params = training_loop_params
        self.simulation_id = simulation_id

    def run_sim(self, save_path: str):
        check_dir(save_path)
        run_log_path = os.path.join(save_path, "run.log")
        error_log_path = os.path.join(save_path, "error.log")
        state_log_path = os.path.join(save_path, "state.log")

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(run_log_path, "w")
        sys.stderr = open(error_log_path, "w")

        # Saving simulation parameters
        print("---> Saving simulation parameters")
        self._update_log_state("saving_params", state_log_path)
        self.save_params(save_path)

        # Initializing models
        print("---> Initializing models")
        self._update_log_state("initializing_models", state_log_path)
        agent, env = self._init_models()

        # Training loop
        print("---> Training loop")
        self._update_log_state("training_loop", state_log_path)
        trained_agent, trained_env = self.training_loop(agent, env, **self.training_loop_params)

        # Saving models
        print("---> Saving models")
        self._update_log_state("saving_models", state_log_path)
        self._save_models(save_path, trained_agent, trained_env)

        print("---> Simulation finished")
        self._update_log_state("successful_run", state_log_path)

        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    def _init_models(self):
        agent = self.agent_class(**self.agent_params)
        env = self.env_class(**self.env_params)
        return agent, env

    def _save_models(self, save_path: str, agent: AgentCore, env: Environment):
        agent.save_agent(os.path.join(save_path, "agent"))
        env.save_environment(os.path.join(save_path, "arena"))

    def save_params(self, save_path: str):
        save_path = os.path.join(save_path, "params.sim")
        pickle.dump(self.__dict__, open(save_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    def load_params(self, load_path: str):
        self.__dict__ = pd.read_pickle(load_path)

    def _update_log_state(self, message: str, save_path: str):
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
