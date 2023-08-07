import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import torch
import importlib

from neuralplayground.agents import Whittington2020
from neuralplayground.arenas import BatchEnvironment
from neuralplayground.arenas import DiscreteObjectEnvironment
from neuralplayground.backend import tem_plotting_loop
from neuralplayground.backend import PlotSim
import neuralplayground.agents.whittington_2020_extras.whittington_2020_analyse as analyse
from neuralplayground.agents.whittington_2020_extras import whittington_2020_parameters as parameters
from neuralplayground.experiments import Sargolini2006Data

simulation_id = "TEM_custom_plot_sim"
save_path = "NeuralPlayground/examples/agent_examples/results_sim/"
training_dict = pd.read_pickle(os.path.join(save_path, "params.dict"))
model_weights = pd.read_pickle(os.path.join(save_path, "agent"))
model_spec = importlib.util.spec_from_file_location("model", save_path + "whittington_2020_model.py")
model = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model)
params = pd.read_pickle(os.path.join(save_path, "agent_hyper"))
tem = model.Model(params)
tem.load_state_dict(model_weights)
tem.eval()

plotting_loop_params = {"n_episode": 50}
sim = PlotSim(simulation_id = simulation_id,
                agent_class = training_dict["agent_class"],
                agent_params = training_dict["agent_params"],
                env_class = training_dict["env_class"],
                env_params = training_dict["env_params"],
                plotting_loop = tem_plotting_loop,
                plotting_loop_params = plotting_loop_params)
print(sim)
sim.plot_sim(save_path)

# Load environments and model_input using pickle
with open(os.path.join(save_path, "NPG_environments.pkl"), "rb") as f:
    environments = pickle.load(f)
with open(os.path.join(save_path, "NPG_model_input.pkl"), "rb") as f:
    model_input = pickle.load(f)

training_dict["params"] = training_dict["agent_params"]
del training_dict["agent_params"]
agent = training_dict["agent_class"](**training_dict["params"])
with torch.no_grad():
    forward = tem(model_input, prev_iter=None)
include_stay_still = False
shiny_envs = [False, False, False, False]
env_to_plot = 0
envs_to_avg = shiny_envs if shiny_envs[env_to_plot] else [not shiny_env for shiny_env in shiny_envs]

correct_model, correct_node, correct_edge = analyse.compare_to_agents(forward, tem, environments, include_stay_still=include_stay_still)
zero_shot = analyse.zero_shot(forward, tem, environments, include_stay_still=include_stay_still)
occupation = analyse.location_occupation(forward, tem, environments)
g, p = analyse.rate_map(forward, tem, environments)
from_acc, to_acc = analyse.location_accuracy(forward, tem, environments)

# Plot rate maps for grid or place cells
agent.plot_rate_map(g)
plt.show()