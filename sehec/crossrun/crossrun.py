from ..models.modelcore import NeuralResponseModel
from ..utils import inheritors

class CrossRun(object):

    def __init__(self):
        self._list_available_models()

    def _list_available_models(self):
