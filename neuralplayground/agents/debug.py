import matplotlib.pyplot as plt
from IPython.display import HTML, display
from tqdm.notebook import tqdm

from neuralplayground.agents import Stachenfeld2018
from neuralplayground.arenas import Hafting2008

display(HTML("<style>.container { width:80% !important; }</style>"))
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from neuralplayground.arenas import Hafting2008

env = Hafting2008(time_step_size=0.1, agent_step_size=None, use_behavioral_data=True)

agent_step_size = 10
discount = 0.9
threshold = 1e-6
lr_td = 1e-2
t_episode = 1000
n_episode = 100
state_density = 1 / agent_step_size
twoDvalue = True

agent = Stachenfeld2018(
    discount=discount,
    t_episode=t_episode,
    n_episode=n_episode,
    threshold=threshold,
    lr_td=lr_td,
    room_width=env.room_width,
    room_depth=env.room_depth,
    state_density=state_density,
    twoD=twoDvalue,
)


plot_every = 100000
total_iters = 0
obs, state = env.reset()
obs = obs[:2]
for i in tqdm(range(100001)):
    # Observe to choose an action
    action = agent.act(obs)  # the action is link to density of state to make sure we always land in a new
    K = agent.update()
    obs, state, reward = env.step(action)
    obs = obs[:2]
    total_iters += 1
    if total_iters % plot_every == 0:
        agent.plot_rate_map(sr_matrix=agent.srmat, eigen_vectors=[1, 10, 15, 20], save_path="./sr_Hating.png")
agent.plot_rate_map(sr_matrix=agent.srmat, eigen_vectors=[1, 10, 15, 20], save_path="./sr_Hating.png")
T = agent.get_T_from_M(agent.srmat_ground)
agent.plot_transition()
T = agent.get_T_from_M(agent.srmat_ground)
agent.plot_transition()
ax = env.plot_trajectory(plot_every=100)
plt.show()
print("hello")
