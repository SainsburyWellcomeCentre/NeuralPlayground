import matplotlib.pyplot as plt
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, TableColumn, DataTable
from bokeh.io import show
from bokeh.models.widgets import DataTable, TableColumn, HTMLTemplateFormatter, Div
import pandas as pd
import os
import glob
import numpy as np


def get_html_formatter(my_col):
    template = """
        <div style="background:<%= 
            (function colorfromint(){
                if(result_col == 'Positive'){
                    return('#f14e08')}
                else if (result_col == 'Negative')
                    {return('#8a9f42')}
                else if (result_col == 'Invalid')
                    {return('#8f6b31')}
                }()) %>; 
            color: white"> 
        <%= value %>
        </div>
    """.replace('result_col', my_col)

    return HTMLTemplateFormatter(template=template)


def get_html_results(results_list):
    def get_block(my_color):
        return '<span style="color:{};font-size:18pt;text-shadow: 1px 1px 2px #000000;">&#9632;</span>'.format(my_color)

    dict_color = {'Failed': '#EA3E19', 'Finished': '#1CEA19', 'Queued': '#199BEA'}

    html_string = ""

    for r in results_list:
        html_string += get_block(dict_color[r])

    return html_string


def model_summary(results_path, config_file, fig_kwargs):

    status_dict = {
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
            status_dict["models"].append(model_conf.config_id)
            for exp_key, exp_conf in model_conf.__dict__.items():
                if "exp" in exp_key:
                    exp_dir_path = os.path.join(model_dir_path, exp_conf.config_id)
                    status_dict["experiments"].append(exp_conf.config_id)
                    for sub_exp_key, sub_exp_conf in exp_conf.__dict__.items():
                        if "sub_exp" in sub_exp_key:
                            status_dict["sub experiments"].append(sub_exp_conf.config_id)
                            sub_exp_dir_path = os.path.join(exp_dir_path, sub_exp_conf.config_id)
                            save_path = os.path.join(results_path, sub_exp_dir_path)
                            run_list = glob.glob(os.path.join(save_path, "run*"))
                            print(run_list, save_path)
                            run_list.sort()
                            for i, run in enumerate(run_list):
                                status_dict["run"].append(i+1)
                                with open(os.path.join(run, "status.log")) as f:
                                    status = f.readline()
                                status_dict["status"].append(status)
                                status_dict["logs"].append(open(os.path.join(run, "run_output.log"), "r"))
                                status_dict["errors"].append(open(os.path.join(run, "err_out.log"), "r"))

    run1 = status_dict["status"][:5]
    run2 = status_dict["status"][5:]
    df_results = pd.DataFrame({
        "Models": status_dict["models"],
        "Experiments": status_dict["experiments"][:len(status_dict["models"])],
        "Sub_exp": status_dict["sub experiments"][:len(status_dict["models"])]
    })

    df_results["Results"] = [get_html_results(run1), get_html_results(run2)]

    source = ColumnDataSource(df_results[["Models", "Experiments", "Sub_exp", "Results"]])
    formatter = HTMLTemplateFormatter()
    columns =[
        TableColumn(field="Models", title="Models"),
        TableColumn(field="Experiments", title="Experiment"),
        TableColumn(field="Sub_exp", title="Sub experiment"),
        TableColumn(field="Results", title="Runs", formatter=formatter)
    ]
    my_table = DataTable(source=source, columns=columns, **fig_kwargs)

    return my_table


def grid_cell_comparison():
    pass


def training_curves(run_path, fig_kwargs):
    model = pd.read_pickle(os.path.join(run_path, "model.pickle"))
    grads = model.grad_history
    p = figure(title="Training", x_axis_label='iters', y_axis_label='grad size',
               width=int(fig_kwargs["width"]*0.5), height=int(fig_kwargs["height"]*0.5))
    p.line(np.arange(len(grads)), grads, line_width=3)
    return p


def foraging_plot(run_path, fig_kwargs):
    env = pd.read_pickle(os.path.join(run_path, "env.pickle"))
    f, ax = env.plot_trajectory(return_figure=True)
    fig_path = os.path.join(run_path, "Foraging_plot.png")
    plt.savefig(fig_path, bbox_inches="tight")
    div_image = Div(
        text='<img src="'+fig_path+'" alt="div_image" style="height: 100%; width: 100%;" >',
        width=int(fig_kwargs["width"]*0.5), height=int(fig_kwargs["height"]*0.5))
    return div_image