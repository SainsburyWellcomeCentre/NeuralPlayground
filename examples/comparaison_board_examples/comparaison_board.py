from sehec.comparison_board.crossrun import CrossRun
from sehec.agent.agent_core import NeuralResponseModel
from sehec.vis.user_interface import ResultsInterface

if __name__ == "__main__":
    results_path = "./test_crossrun_results"
    crossrun = CrossRun(results_path=results_path, reload_experiment=True)
    # crossrun.run_cross_exp()

    interface = ResultsInterface(results_path=results_path)
    interface.generate_panel()
