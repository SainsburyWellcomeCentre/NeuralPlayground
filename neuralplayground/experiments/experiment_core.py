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
