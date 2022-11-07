import neuralplayground
from neuralplayground.experimentconfig import cfg
from neuralplayground.comparison_board.default_run import *
import os
import glob
import shutil
import pickle


class CrossRun(object):

    def __init__(self, config_file=None, results_path="../results", reload_experiment=True):
        print("Init cross run")
        if config_file is None:
            self.config_file = cfg
        else:
            self.config_file = config_file
        self._print_tree()
        self.results_path = results_path
        self.pckg_path = neuralplayground.__path__[0]
        if not reload_experiment:
            self.go_to_experiment(create_experiments=True)

    def _print_tree(self):
        print(self.config_file.get_config_tree())

    def go_to_experiment(self, create_experiments=False, check_experiments=False, run_experiment=False):
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
                                if create_experiments:
                                    print("Reset results from " + save_path)
                                    try:
                                        shutil.rmtree(save_path+"/")
                                    except:
                                        pass
                                    for i in range(sub_exp_conf.n_runs):
                                        now_str = str(datetime.now()).replace(" ", "_").replace(":", "-").replace(".", "_")
                                        now_str = "run_"+str(i+1)+"_"+now_str
                                        run_save_path = os.path.join(save_path, now_str)
                                        check_experiment_status(dir=run_save_path, create=True)
                                        pickle.dump(sub_exp_conf,
                                                    open(os.path.join(run_save_path, "params.cfg"), "wb"),
                                                    pickle.HIGHEST_PROTOCOL)
                                if check_experiments:
                                    run_list = glob.glob(os.path.join(save_path, "run*"))
                                    for run_dir in run_list:
                                        check_experiment_status(run_dir, check=True)
                                if run_experiment:
                                    run_list = glob.glob(os.path.join(save_path, "run*"))
                                    for run_dir in run_list:
                                        check_experiment_status(run_dir, check=True)
                                        log_path = os.path.join(run_dir, "run_output.log")
                                        script_path = os.path.join(self.pckg_path, "crossrun/default_routine_script.py")
                                        cmd_line = 'python -u '+script_path+' "'+run_dir+'" > '+log_path
                                        print(cmd_line)
                                        exit_status = os.system(cmd_line)
                                        with open(os.path.join(run_dir, 'status.log'), 'w') as f:
                                            if exit_status == 0:
                                                f.write('Finished')
                                            else:
                                                f.write('Failed')
                                        # print("EXIT STATUS", exit_status)

    def run_cross_exp(self):
        self.go_to_experiment(run_experiment=True)


def check_experiment_status(dir, create=False, check=False):
    check = os.path.isdir(dir)
    if create:
        os.makedirs(dir)
        with open(os.path.join(dir, 'status.log'), 'w') as f:
            f.write('Queued')
    return check
