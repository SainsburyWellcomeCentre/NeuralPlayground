from neuralplayground.backend import SimulationManager
from neuralplayground.backend.default_simulation import (
    coreAgent_test_1,
    coreAgent_test_2,
    coreAgent_test_3,
    coreAgent_test_4,
    coreAgent_test_5,
    coreAgentEnv_test,
    coreAgentEnv_test_2,
    coreEnv_test_1,
    coreEnv_test_2,
    coreEnv_test_3,
    coreEnv_test_4,
    stachenfeld_in_2d,
    stachenfeld_in_hafting,
    stachenfeld_in_merging_room_1,
    stachenfeld_in_merging_room_2,
    stachenfeld_in_sargolini,
    stachenfeld_in_wernle_1,
    stachenfeld_in_wernle_2,
    weber_in_2d,
    weber_in_hafting,
    weber_in_merging_room,
    weber_in_sargolini,
    weber_in_wernle,
)


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

    # sim_object2.run_sim("try_results_sim4/")

    sim_manager = SimulationManager(
        [
            weber_in_wernle,
            weber_in_merging_room,
            weber_in_sargolini,
            weber_in_hafting,
            weber_in_2d,
            stachenfeld_in_wernle_1,
            stachenfeld_in_wernle_2,
            stachenfeld_in_merging_room_1,
            stachenfeld_in_merging_room_2,
            stachenfeld_in_sargolini,
            stachenfeld_in_hafting,
            stachenfeld_in_2d,
            coreEnv_test_1,
            coreEnv_test_2,
            coreEnv_test_3,
            coreEnv_test_4,
            coreAgentEnv_test,
            coreAgentEnv_test_2,
            coreAgent_test_1,
            coreAgent_test_2,
            coreAgent_test_3,
            coreAgent_test_4,
            coreAgent_test_5,
        ],
        runs_per_sim=2,
        manager_id="test_dev",
        verbose=True,
    )

    sim_manager.generate_sim_paths()

    sim_manager.run_all()

    sim_manager.check_run_status()

    # aux_manager_2 = SimulationManager(existing_simulation="test_dev")
    # print(aux_manager_2)
    # aux_manager_2.check_run_status()


if __name__ == "__main__":
    main()
