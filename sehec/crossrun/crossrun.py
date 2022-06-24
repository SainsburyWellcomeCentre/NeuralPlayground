from ..experimentconfig import cfg
from sehec.crossrun.default_run import *
from sehec.utils import check_directory
import os


class CrossRun(object):

    def __init__(self, config_file=None, results_path="../results"):
        print("Init cross run")
        if config_file is None:
            self.config_file = cfg
        else:
            self.config_file = config_file
        self._print_tree()
        self.results_path = results_path
        check_directory(self.results_path)

    def _print_tree(self):
        print(self.config_file.get_config_tree())

    def go_to_experiment(self, set_directories=True, run_experiment=False):
        for model_key, model_conf in self.config_file.__dict__.items():
            if "model" in model_key:
                model_dir_path = model_conf.config_id
                for exp_key, exp_conf in model_conf.__dict__.items():
                    if "exp" in exp_key:
                        exp_dir_path = os.path.join(model_dir_path, exp_conf.config_id)
                        for sub_exp_key, sub_exp_conf in exp_conf.__dict__.items():
                            if "sub_exp" in sub_exp_key:
                                sub_exp_dir_path = os.path.join(exp_dir_path, sub_exp_conf.config_id)
                                save_path = os.path.join(self.results_path, sub_exp_dir_path)
                                for i in range(sub_exp_conf.n_runs):
                                    now_str = str(datetime.now()).replace(" ", "_").replace(":", "-").replace(".", "_")
                                    now_str = "run_"+str(i+1)+"_"+now_str
                                    run_save_path = os.path.join(save_path, now_str)

                                    check_directory(run_save_path)
                                    print(save_path)
                                    default_run(sub_exp_config=sub_exp_conf, save_path=save_path)
