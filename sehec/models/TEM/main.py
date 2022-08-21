import matplotlib.pyplot as plt

from sehec.envs.arenas.TEMenv import TEMenv
from sehec.models.TEM.model import *
from sehec.models.TEM.parameters import *

gen_path, train_path, model_path, save_path, script_path = make_directories()

pars = default_params()
save_params(pars, save_path, script_path)
pars_orig = pars.copy()

# Initialise Environment(s) and Agent (variables, weights etc.)
print("Initialising Graph")
env_name = "TEMenv"
mod_name = "TEM"

envs = TEMenv(environment_name=env_name, **pars)
agent = TEM(model_name=mod_name, **pars)
print("Graph Initialised")

# Run Model
it = 0
print("Start Training")
for i in range(pars['n_iters']):
    for j in range(pars['n_episode']):
        # RL Loop
        obs, states, rewards, actions, direcs = envs.step(agent.act)
        x_, p, g = agent.update(obs, direcs, it, j)
        it += 1
        print("finished episode ", j)
    print("finished iteration ", i)

envs.plot_trajectory()
plt.show()
