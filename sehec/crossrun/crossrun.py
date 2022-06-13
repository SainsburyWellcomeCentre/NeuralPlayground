from ..models.modelcore import NeuralResponseModel
from ..experimentconfig import cfg


class CrossRun(object):

    def __init__(self, config_file=None):
        print("Init cross run")
        if config_file is None:
            self.config_file = cfg
        else:
            self.config_file = config_file
        self._list_available_models()

    def _list_available_models(self):
        print("Models to run: ")
        for key in self.config_file.available_params:
            if "model" in key:
                print(self.config_file.__dict__[key].config_id)
