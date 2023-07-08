import os

from neuralplayground.backend.default_experiment import SingleSim, sim_object


def main():
    results_path = "try_results_sim/"
    print(sim_object)
    sim_object.run_sim(results_path)

    print("you should see this in the terminal")
    aux_sim = SingleSim()
    print(aux_sim)
    aux_sim.load_params(os.path.join(results_path, "params.sim"))
    print(aux_sim)
    aux_sim.run_sim("try_results_sim2/")


if __name__ == "__main__":
    main()
