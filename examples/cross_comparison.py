from sehec.crossrun.crossrun import CrossRun
from sehec.models.modelcore import NeuralResponseModel

if __name__ == "__main__":
    results_path = "./test_crossrun_results"
    crossrun = CrossRun(results_path=results_path)
    crossrun.run_comparisons()