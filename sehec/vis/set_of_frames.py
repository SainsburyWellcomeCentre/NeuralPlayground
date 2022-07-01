from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, TableColumn, DataTable
from bokeh.io import show
import pandas as pd
import os
import glob


def model_summary(results_path, config_file):

    status_df = {
        "models": [],
        "experiments": [],
        "sub experiments": [],
        "run": [],
        "status": [],
        "logs": [],
        "errors": []
    }

    for model_key, model_conf in config_file.__dict__.items():
        if "model" in model_key:
            model_dir_path = model_conf.config_id
            status_df["models"].append(model_conf.config_id)
            for exp_key, exp_conf in model_conf.__dict__.items():
                if "exp" in exp_key:
                    exp_dir_path = os.path.join(model_dir_path, exp_conf.config_id)
                    status_df["experiments"].append(exp_conf.config_id)
                    for sub_exp_key, sub_exp_conf in exp_conf.__dict__.items():
                        if "sub_exp" in sub_exp_key:
                            status_df["sub experiments"].append(sub_exp_conf.config_id)
                            sub_exp_dir_path = os.path.join(exp_dir_path, sub_exp_conf.config_id)
                            save_path = os.path.join(results_path, sub_exp_dir_path)
                            run_list = glob.glob(os.path.join(save_path, "run*"))
                            print(run_list, save_path)
                            run_list.sort()
                            for i, run in enumerate(run_list):
                                status_df["run"].append(i+1)
                                with open(os.path.join(run, "status.log")) as f:
                                    status = f.readline()
                                status_df["status"].append(status)
                                status_df["logs"].append(open(os.path.join(run, "run_output.log"), "r"))
                                status_df["errors"].append(open(os.path.join(run, "err_out.log"), "r"))

    print(status_df)