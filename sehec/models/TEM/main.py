import matplotlib.pyplot as plt

from sehec.envs.arenas.TEMenv import TEMenv
from sehec.models.TEM.model import TEM
from sehec.models.TEM.parameters import *


pars = default_params()

env_name = "TEMenv"
mod_name = "TEM"

# Initialise Environment(s) and Agent (variables, weights etc.)
envs = TEMenv(environment_name=env_name, **pars)
agent = TEM(model_name=mod_name, **pars)

for i in range(pars['n_iters']):
    for j in range(pars['n_episode']):
        # RL Loop
        obs, states, rewards, actions, direcs = envs.step(agent.act)
        x_, p, g = agent.update(direcs, obs, j)
        print("finished episode ", j)
    print("finished iteration ", i)

envs.plot_trajectory()
plt.show()
