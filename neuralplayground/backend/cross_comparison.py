import os

from neuralplayground.utils import check_dir


class SimulationsManager(object):
    """Class to manage the runs of multiple combinations of agents, environments, parameters and training loops"""

    def __init__(
        self, simulation_list, runs_per_sim: int, simulation_id: str = "default_comparison", results_path: str = "results"
    ):
        self.simulation_list = simulation_list
        self.runs_per_sim = runs_per_sim
        self.simulation_id = simulation_id
        self.results_path = results_path
        self._generate_sim_paths()

    def _generate_sim_paths(self):
        """Generate the paths for the simulations
        If the path does not exist, it will be created
        if the path exists, it will save the results in the existing path
        """
        self.full_results_path = os.path.join(self.results_path, self.simulation_id)
        self.simulation_paths = []
        for sim in self.simulation_list:
            sim_path = os.path.join(self.full_results_path, sim.simulation_id)
            check_dir(sim_path)
            self.simulation_paths.append(sim_path)

    def run_all(self):
        for sim in self.simulation_list:
            for run in range(self.runs_per_sim):
                sim.run_sim()
                sim.save_sim()

    def run_single_sim(self, sim_index):
        for run in range(self.runs_per_sim):
            current_sim = self.simulation_list[sim_index]
            current_sim.initialize_models()
            current_sim.run_sim()
            current_sim.save_sim()

    def check_runs(self):
        pass

    def set_config(self):
        pass


class SingleSim(object):
    def __init__(
        self, agent_class, agent_params, env_class, env_params, training_loop, training_loop_params, simulation_id: str
    ):
        self.agent_class = agent_class
        self.agent_params = agent_params
        self.env_class = env_class
        self.env_params = env_params
        self.training_loop = training_loop
        self.training_loop_params = training_loop_params
        self.simulation_id = simulation_id

    def run_sim(self, save_path: str):
        agent = self.agent_class(**self.agent_params)
        env = self.env_class(**self.env_params)
        trained_agent, trained_env = self.training_loop(agent, env, **self.training_loop_params)

    def save_sim(self, save_path: str):
        self.agent.save_agent()
        self.env.save_environment()
