from neuralplayground.datasets import fetch_data_path


class Experiment(object):
    """Main abstract experiment class, created just for consistency purposes"""

    def __init__(self, experiment_name: str = "abstract_experiment", data_url: str = None, paper_url: str = None):
        """Constructor for the abstract Experiment class

        Parameters
        ----------
        experiment_name: str
            Name of the experiment
        data_url: str
            URL to the data used in the experiment, make sure it is publicly available for usage and download
        paper_url: str
            URL to the paper describing the experiment
        """
        self.experiment_name = experiment_name
        self.data_url = data_url
        self.paper_url = paper_url

    def _find_data_path(self, data_path: str = None):
        """Fetch data from NeuralPlayground data repository
        if no data path is supplied by the user"""
        if data_path is None:
            self.data_path = fetch_data_path("hafting_2008")
        else:
            self.data_path = data_path
