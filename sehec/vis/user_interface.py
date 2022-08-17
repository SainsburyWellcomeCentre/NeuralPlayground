import sehec
from sehec.experimentconfig import cfg
from bokeh.models import Panel, Tabs
from bokeh.plotting import figure, gridplot
from bokeh.layouts import row, column
from bokeh.io import show
import bokeh
import os
import glob
from sehec.vis.set_of_frames import model_summary, foraging_plot, training_curves, grid_cell_comparison


class ResultsInterface(object):

    def __init__(self, interface_path="../interface", results_path="../results",
                 config_file=None):
        self.results_path = results_path
        if config_file is None:
            self.config_file = cfg
        else:
            self.config_file = config_file
        self.pckg_path = sehec.__path__[0]
        self._default_fig_params()

    def _default_fig_params(self):
        """ Default figure parameters """
        self.figure_kwargs = dict()
        self.figure_kwargs["width"] = 1200
        self.figure_kwargs["height"] = 1000

    def generate_panel(self):
        """ Loop to generate main user interface """

        """ Models summary, run status and results summary """
        self.model_summary_fig = model_summary(results_path=self.results_path,
                                               config_file=self.config_file,
                                               fig_kwargs=self.figure_kwargs)
        self.model_summary_panel = Panel(child=self.model_summary_fig, title="Models summary")

        """ Initializing model panels """
        self.model_figs = {}
        self.model_panels = {}
        self.model_experiments = {}

        for model_key, model_conf in self.config_file.__dict__.items():
            if "model" in model_key:
                model_dir_path = model_conf.config_id

                """ Experiment summary panel """
                self.exp_summary_fig = figure(**self.figure_kwargs)
                self.exp_summary_panel = Panel(child=self.exp_summary_fig, title="Experiments summary")

                """ Creating sub panel for experiments """
                self.model_experiments[model_key] = {}

                """ Going to sub experiments """
                for exp_key, exp_conf in model_conf.__dict__.items():
                    if "exp" in exp_key:
                        exp_dir_path = os.path.join(model_dir_path, exp_conf.config_id)
                        """ Last panel/tab level with experiments """
                        self.model_experiments[model_key][exp_key] = {}
                        for i, (sub_exp_key, sub_exp_conf) in enumerate(exp_conf.__dict__.items()):
                            if "sub_exp" in sub_exp_key:
                                sub_exp_dir_path = os.path.join(exp_dir_path, sub_exp_conf.config_id)
                                save_path = os.path.join(self.results_path, sub_exp_dir_path)
                                run_list = glob.glob(os.path.join(save_path, "run*"))

                        fig1 = foraging_plot(run_list[0], fig_kwargs=self.figure_kwargs)
                        fig2 = training_curves(run_list[0], fig_kwargs=self.figure_kwargs)
                        exp_figure = row([fig1, fig2])
                        if "grid_cell_comparison" in exp_conf.sub_exp_1.list_of_plots:
                            grid_cell_figure = grid_cell_comparison(run_list[0], fig_kwargs=self.figure_kwargs)
                            exp_figure = column([exp_figure, grid_cell_figure])
                        self.model_experiments[model_key][exp_key] = Panel(
                            child=exp_figure,
                            title=exp_conf.config_id
                        )

                """ Creating specific model panel """
                self.model_figs[model_key] = Tabs(tabs=[self.exp_summary_panel, ]+list(self.model_experiments[model_key].values()))
                self.model_panels[model_key] = Panel(child=self.model_figs[model_key], title=model_conf.config_id)

        self.main_fig = Tabs(tabs=[self.model_summary_panel, ]+list(self.model_panels.values()))
        show(self.main_fig)

