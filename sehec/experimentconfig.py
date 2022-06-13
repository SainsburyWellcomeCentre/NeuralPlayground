#  Config file with default parameters for each experiment
import numpy as np


class Config(object):
    """ Config object """
    def __init__(self, config_id, **kwargs):
        self.config_id = config_id
        self.__dict__.update(kwargs)
        self.available_params = list(self.__dict__.keys())

    def remove_attribute(self, attr):
        delattr(self, attr)

"""
Config files have 4 levels
1- Model
    2- Exp
        3- Sub Experiment
            4.1 - Model params
            4.2 - Environment params
To have access to params, import this file as cfg, then use
cfg.model.exp.subexp.model_params or cfg.model.exp.subexp.env_params

For details on parameter definitions for environments or models, see source code
"""


""" 1 - Weber and Sprekeler """
model_id = "weber_and_sprekeler"

""" 2 - Sargolini 2D Room """
sargolini_id = "sargolini2006"

""" 3 - 2D Foraging in sargolini """
sub_exp_id = "sargolini2006_2dforaging"

""" 4 - Environment parameters """
env_params = Config(config_id="Sargolini2006_params",
                    data_path="Sargolini2006",
                    environment_name="Sargolini2006_2D")
model_params = Config(config_id="weber_and_sprekeler_params",
                      exc_eta=2e-4,
                      inh_eta=8e-4,
                      model_name="WeberAndSprekeler",
                      sigma_exc=np.array([0.05, 0.05]),
                      sigma_inh=np.array([0.1, 0.1]),
                      Ne=4900,
                      Ni=1225,
                      Nef=1,
                      Nif=1,
                      alpha_i=1,
                      alpha_e=1,
                      we_init=1.0,
                      wi_init=1.5,
                      agent_step_size=0.1)

sub_exp_1 = Config(config_id=sub_exp_id,
                   env_params=env_params,
                   model_params=model_params)

sargolini2006 = Config(config_id=sargolini_id,
                       sub_exp_1=sub_exp_1)

weber_and_sprekeler = Config(config_id=model_id,
                             exp_1=sargolini2006)

""" 1 - SR """

cfg = Config(config_id="full_configuration",
             model1=weber_and_sprekeler)


if __name__ == "__main__":
    params = {"config_id": "model1",
              "param1": 30,
              "param2": "bla"}
    exp_config = Config(**params)
    print("debug")