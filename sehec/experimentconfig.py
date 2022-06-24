#  Config file with default parameters for each experiment
import numpy as np
from copy import deepcopy


class Config(object):
    """ Config object """
    def __init__(self, config_id, **kwargs):
        self.config_id = config_id
        self.__dict__.update(kwargs)
        self.available_params = list(self.__dict__.keys())

    def remove_attribute(self, attr):
        delattr(self, attr)

    def get_config_tree(self, level=0):
        config_mssg = self.config_id + "\n"
        sub_conf_mssg = ""
        for key, val in self.__dict__.items():
            if key == "config_id" or key == "available_params":
                continue
            else:
                if isinstance(val, Config):
                    sub_conf_mssg += "--> "*(level+1) + key + ": " + val.get_config_tree(level+1) + "\n"
                else:
                    config_mssg += "--> " * (level + 1) + key + ": " + str(val) + ", type: "+str(type(val))+"\n"
        if len(sub_conf_mssg) != 0:
            config_mssg += "\n"
        config_mssg += sub_conf_mssg
        return config_mssg

    def copy(self):
        return deepcopy(self)

    def add_attribute(self, attr, val):
        setattr(self, attr, val)

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
sub_exp_id = "sargolini2006_2d_foraging"

""" 4 - Environment parameters """
env_params = Config(config_id="Sargolini2006_params",
                    class_name="BasicSargolini2006",
                    data_path="../envs/experiments/Sargolini2006",
                    environment_name="Sargolini2006",
                    time_step_size=0.1)
model_params = Config(config_id="weber_and_sprekeler_params",
                      class_name="ExcInhPlasticity",
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
                      agent_step_size=0.1,
                      ro=1,
                      room_width=10,
                      room_depth=10,
                      n_iters=100)

sub_exp_1 = Config(config_id=sub_exp_id,
                   env_params=env_params,
                   model_params=model_params,
                   n_runs=3)

sargolini2006 = Config(config_id=sargolini_id,
                       sub_exp_1=sub_exp_1)

weber_and_sprekeler = Config(config_id=model_id,
                             exp_1=sargolini2006)

""" 1 - SR """

cfg = Config(config_id="### Full Configuration ###",
             model1=weber_and_sprekeler)

custom_classes = Config(config_id="Custom_class_paths",
                        custom_classes_path=["from sehec.models.SRKim import SR",])

if __name__ == "__main__":

    print(cfg.config_tree())