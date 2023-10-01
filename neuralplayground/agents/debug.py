import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from tqdm.notebook import tqdm
from neuralplayground.arenas import Simple2D, MergingRoom, Sargolini2006, Hafting2008, BasicSargolini2006,Wernle2018
from neuralplayground.utils import create_circular_wall
from neuralplayground.agents import  Stachenfeld2018
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from tqdm.notebook import tqdm
from neuralplayground.arenas import Simple2D, MergingRoom, Sargolini2006, Hafting2008, BasicSargolini2006, Wernle2018
from neuralplayground.utils import create_circular_wall
from neuralplayground.agents import Stachenfeld2018
from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))
import sys
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
from neuralplayground.arenas import Simple2D, MergingRoom, Sargolini2006, Hafting2008, BasicSargolini2006,Wernle2018
from neuralplayground.agents import Weber2018, RandomAgent, LevyFlightAgent
from neuralplayground.experiments import Wernle2018Data
from neuralplayground.arenas import Simple2D,Wernle2018,  MergingRoom

env = Hafting2008(time_step_size=0.1,
                  agent_step_size=None,
                  use_behavioral_data=True)

agent_step_size = 10
discount = .9
threshold = 1e-6
lr_td = 1e-2
t_episode = 1000
n_episode = 100
state_density = (1 / agent_step_size)
twoDvalue = True

agent = Stachenfeld2018(discount=discount, t_episode=t_episode, n_episode=n_episode, threshold=threshold, lr_td=lr_td,
               room_width=env.room_width, room_depth=env.room_depth, state_density=state_density, twoD=twoDvalue)


plot_every = 100000
total_iters = 0
obs, state = env.reset()
obs = obs[:2]
for i in tqdm(range(100001)):
# Observe to choose an action
    action = agent.act(obs)  # the action is link to density of state to make sure we always land in a new
    K  = agent.update()
    obs, state, reward = env.step(action)
    obs= obs[:2]
    total_iters += 1
    if total_iters % plot_every == 0:
        agent.plot_rate_map(sr_matrix=agent.srmat,eigen_vectors=[1,10,15,20], save_path='./sr_Hating.png')
agent.plot_rate_map(sr_matrix=agent.srmat,eigen_vectors=[1,10,15,20], save_path='./sr_Hating.png')
T = agent.get_T_from_M(agent.srmat_ground)
agent.plot_transition()
T = agent.get_T_from_M(agent.srmat_ground)
agent.plot_transition()
ax = env.plot_trajectory(plot_every=100)
plt.show()
print('hello')