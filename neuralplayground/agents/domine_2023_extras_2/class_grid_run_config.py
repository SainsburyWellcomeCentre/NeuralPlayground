from typing import Dict, Union

from neuralplayground.agents.domine_2023_extras_2.class_config_template import (
    ConfigTemplate,
)
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
