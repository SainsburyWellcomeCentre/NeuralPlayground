from sehec.crossrun.crossrun import CrossRun
from sehec.models.modelcore import NeuralResponseModel
from sehec.vis.user_interface import ResultsInterface

if __name__ == "__main__":
    results_path = "./test_crossrun_results"
    #crossrun = CrossRun(results_path=results_path, reload_experiment=False)
    #crossrun.run_cross_exp()

    interface = ResultsInterface(results_path=results_path)
    interface.generate_panel()
