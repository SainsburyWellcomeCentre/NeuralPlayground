import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sehec.envs.arenas.simple2d import Simple2D, Sargolini2006, Hafting2008,BasicSargolini2006
from sehec.models.SRKim import SR

run_raw_data = True


if run_raw_data == False:
    room_width = 7
    room_depth = 7
    env_name = "env_example"
    time_step_size = 1  # seg
    agent_step_size = 1

    # Init environment
    env = Simple2D(environment_name=env_name,
                   room_width=room_width,
                   room_depth=room_depth,
                   time_step_size=time_step_size,
                   agent_step_size=agent_step_size)

    discount = .9
    threshold = 1e-6
    lr_td = 1e-2
    t_episode = 100
    n_episode = 10000
    state_density = int(1 / agent_step_size)
    twoDvalue=True
    agent = SR(discount=discount, t_episode=t_episode, n_episode=n_episode, threshold=threshold, lr_td=lr_td,
               room_width=env.room_width, room_depth=env.room_depth, state_density=state_density,twoD=twoDvalue)

    # Choose your function depending on the type of env '2D_env' or '1D_env' + initialisies the smart as well
    # Choose your function depending on the type of env '2D_env' or '1D_env' + initialisies the smart as well
    # Only run if twoDvalue=True
    sr = agent.update_successor_rep()  # Choose your type of Update
    sr_td = agent.update_successor_rep_td_full()  # Choose your type of Update
    sr_sum = agent.successor_rep_sum()
    agent.plot_eigen(sr, save_path=None)
    agent.plot_eigen(sr_sum, save_path=None)
    agent.plot_eigen(sr_td, save_path=None)

    plot_every = 1000
    total_iters = 0
    obs, state = env.reset()
    obs = obs[:2]
    current_state = agent.obs_to_state(obs)
    for i in tqdm(range(n_episode*t_episode)):
            action = agent.act(obs)  # the action is link to density of state to make sure we always land in a new
            obs, state, reward = env.step(action)
            current_state, K = agent.update_successor_rep_td(obs, current_state)
            total_iters += 1
            # if total_iters % plot_every == 0:
            # agent.plot_eigen(K, save_path="./figures/M_processed_iter_" + str(total_iters) + ".pdf")
    T = agent.get_T_from_M(K)
    # agent.plot_trantion(T, save_path="./figures/transtion.pdf")

    for i in tqdm(range(n_episode)):
        for j in range(t_episode):
            action = agent.act(obs)  # the action is link to density of state to make sure we always land in a new
            obs, state, reward = env.step(action)
            current_state,K  = agent.update_successor_rep_td(obs, current_state)
            total_iters += 1
            # if total_iters % plot_every == 0:
                #agent.plot_eigen(K, save_path="./figures/M_processed_iter_" + str(total_iters) + ".pdf")
    T = agent.get_T_from_M(K)
    # agent.plot_trantion(T, save_path="./figures/transtion.pdf")
else:
    data_path = "../sehec/envs/experiments/Sargolini2006/"
    env = BasicSargolini2006(data_path=data_path,
                             time_step_size=0.1,
                             agent_step_size=None)

    agent_step_size = 10
    discount = .9
    threshold = 1e-6
    lr_td = 1e-2
    t_episode = 10
    n_episode = 50
    state_density = (1 / agent_step_size)
    twoDvalue = True

    agent = SR(discount=discount, t_episode=t_episode, n_episode=n_episode, threshold=threshold, lr_td=lr_td,
               room_width=env.room_width, room_depth=env.room_depth, state_density=state_density, twoD=twoDvalue)

    sr = agent.update_successor_rep()  # Choose your type of Update

    #agent.plot_eigen(sr, save_path=None)
    #agent.plot_eigen(sr_sum, save_path="figures/sr_sum.pdf")
    # agent.plot_eigen(sr_td, save_path="./figures/sr_full_td.pdf")

    plot_every = 1000000
    total_iters = 0
    obs, state = env.reset()
    obs = obs[:2]
    for i in tqdm(range(10000000)):
        # Observe to choose an action
        # the action is link to density of state to make sure we always land in a new
        action = agent.act(obs)
        agent.update()
        obs, state, reward = env.step(action)
        obs = obs[:2]
        total_iters += 1


