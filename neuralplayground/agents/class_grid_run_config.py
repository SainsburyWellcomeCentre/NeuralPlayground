from typing import Dict, Union

from class_config_template import ConfigTemplate
from config_manager import base_configuration


class GridConfig(base_configuration.BaseConfiguration):
    def __init__(
        self,
        config: Union[str, Dict],
    ) -> None:
        super().__init__(
            configuration=config,
            template=ConfigTemplate.base_config_template,
        )
        self._validate_configuration()

    def _validate_configuration(self):
        """Method to check for non-trivial associations
        in the configuration.
        """
        pass
        # if self.teacher_configuration == constants.BOTH_ROTATION:
        #     assert self.scale_forward_by_hidden == True, (
        #         "In both rotation regime, i.e. mean field limit, "
        #         "need to scale forward by 1/K."
        #     )
        # else:
        #     assert self.scale_forward_by_hidden == False, (
        #         "When not in both rotation regime, i.e. mean field limit, "
        #         "no need to scale forward by 1/K."
        #     )
