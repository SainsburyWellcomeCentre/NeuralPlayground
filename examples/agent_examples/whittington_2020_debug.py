import importlib
import os

import numpy as np
import pandas as pd

from neuralplayground.plotting import PlotSim

# simulation_id = "examples/agent_examples/TEM_test_with_break"
simulation_id = "TEM_test_witch_break"
save_path = simulation_id + "/"
plotting_loop_params = {"n_walk": 200}

training_dict = pd.read_pickle(os.path.join(os.getcwd(), save_path, "params.dict"))
model_weights = pd.read_pickle(os.path.join(save_path, "agent"))
model_spec = importlib.util.spec_from_file_location("model", save_path + "whittington_2020_model.py")
model = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model)
params = pd.read_pickle(os.path.join(save_path, "agent_hyper"))
tem = model.Model(params)
tem.load_state_dict(model_weights)

sim = PlotSim(
    simulation_id=simulation_id,
    agent_class=training_dict["agent_class"],
    agent_params=training_dict["agent_params"],
    env_class=training_dict["env_class"],
    env_params=training_dict["env_params"],
    plotting_loop_params=plotting_loop_params,
)

trained_agent, trained_env = sim.plot_sim(save_path, random_state=False, custom_state=[0.0, 0.0])
# trained_env.plot_trajectories();

max_steps_per_env = np.random.randint(4000, 5000, size=params["batch_size"])
current_steps = np.zeros(params["batch_size"], dtype=int)

obs, state = trained_env.reset(random_state=False, custom_state=[0.0, 0.0])
for i in range(200):
    while trained_agent.n_walk < params["n_rollout"]:
        actions = trained_agent.batch_act(obs)
        obs, state, reward = trained_env.step(actions, normalize_step=True)
    trained_agent.update()

    current_steps += params["n_rollout"]
    finished_walks = current_steps >= max_steps_per_env
    if any(finished_walks):
        for env_i in np.where(finished_walks)[0]:
            trained_env.reset_env(env_i)
            trained_agent.prev_iter[0].a[env_i] = None

            max_steps_per_env[env_i] = params["n_rollout"] * np.random.randint(
                trained_agent.walk_length_center - params["walk_it_window"] * 0.5,
                trained_agent.walk_length_center + params["walk_it_window"] * 0.5,
            )
            current_steps[env_i] = 0
