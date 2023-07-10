from neuralplayground.backend import SimulationManager
from neuralplayground.backend.default_experiment import sim_object1, sim_object2


def main():
    # results_path = "try_results_sim/"
    # print(sim_object1)
    # sim_object1.run_sim(results_path)
    #
    # print("you should see this in the terminal")
    # aux_sim = SingleSim()
    # print(aux_sim)
    # aux_sim.load_params(os.path.join(results_path, "params.sim"))
    # print(aux_sim)
    # aux_sim.run_sim("try_results_sim2/")
    #
    # print("you should see this in the terminal")
    # sim_object2.run_sim("try_results_sim3/")
    # print(sim_object2)

    sim_manager = SimulationManager([sim_object1, sim_object2], runs_per_sim=5, manager_id="test_dev", verbose=True)

    sim_manager.generate_sim_paths()


if __name__ == "__main__":
    main()
