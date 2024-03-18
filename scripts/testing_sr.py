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


def main():
    room_width = [-6, 6]
    room_depth = [-6, 6]
    env_name = "env_example"
    time_step_size = 0.2
    agent_step_size = 0.5
    # Init environment
    env = Simple2D(environment_name=env_name,
                   arena_x_limits=room_width,
                   arena_y_limits=room_depth,
                   time_step_size=time_step_size,
                   agent_step_size=agent_step_size)
    # Init agent
    discount = .9
    threshold = 1e-6
    lr_td = 0.2
    t_episode = 100  # 1000
    n_episode = 100  # 1000
    state_density = int(1 / agent_step_size)
    agent = Stachenfeld2018(discount=discount, threshold=threshold, lr_td=lr_td,
                            room_width=env.room_width, room_depth=env.room_depth, state_density=state_density,
                            twoD=True, agent_step_size=agent_step_size)

    sr_sum = agent.successor_rep_solution()
    state_to_state = np.reshape(sr_sum, newshape=(agent.width, agent.depth, agent.width, agent.depth))

    # sr_td = agent.update_successor_rep_td_full(300, 300)

    plot_every = 1000
    total_iters = 0
    obs, state = env.reset()
    for i in tqdm(range(n_episode)):
        for j in range(t_episode):
            obs = obs[:2]
            action = agent.act(obs)  # the action is link to density of state to make sure we always land in a new
            obs, new_state, reward = env.step(action)
            agent.update(next_state=obs)
            total_iters += 1

    print(sr_sum)

    plt.imshow(sr_sum)
    plt.show()


if __name__ == "__main__":
    main()