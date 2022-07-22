import matplotlib.pyplot as plt

from sehec.envs.arenas.TEMenv import TEMenv
from sehec.models.TEM.model import TEM
from sehec.models.TEM.parameters import *


pars = default_params()

env_name = "TEMenv"
mod_name = "TEM"

# Initialise Environment(s)
envs = TEMenv(environment_name=env_name, **pars)
agent = TEM(model_name=mod_name, **pars)

for i in range(pars['n_episode']):
    obs, state = envs.reset()

    # Initalise Hebbian Weights
    a_rnn, a_rnn_inv = agent.initialise_hebbian()

    # RL Loop
    # actions, direc = act(obs)
    obs, states, rewards, actions, direcs = envs.step(obs)
    agent.update(direcs, obs, a_rnn, a_rnn_inv)

envs.plot_trajectory()
plt.show()