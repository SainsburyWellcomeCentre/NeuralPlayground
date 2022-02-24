import sys
sys.path.append("../")
import matplotlib.pyplot as plt
import numpy as np
from simple2d import Simple2D

class MergingRoom2D(Simple2D):
    def __init__(self, environment_name="MergingRoom", **env_kwargs):
        self.environment_name = environment_name
        self.arena_limits = np.array([[0, self.room_width],
                                      [-self.room_depth/2, self.room_depth/2]])
        limits = np.abs(np.diff(self.arena_limits, axis=1))
        self.room_width = limits[0, 0]
        self.room_depth = limits[1, 0]
        env_kwargs["room_width"] = self.room_width
        env_kwargs["room_depth"] = self.room_depth
        env_kwargs["agent_step_size"] = 1/50                              
        super().__init__(environment_name, **env_kwargs)
        self.metadata["doi"] = "https://doi.org/10.11582/2017.00023"

class RandomAgent(object):
    
    def act(self, observation):
        return np.random.normal(scale=0.1, size=(2,))

if __name__ == "__main__":
    env_name = ["merging_room_envA", "merging_room_envA", "merging_room_envA"]
    room_width = 200
    room_depth = [100, 100, 200]
    time_step_size = 0.1
    agent_step_size = 0.5

    envA = MergingRoom2D(environment_name=env_name[0],
               room_width = room_width,
               room_depth = room_depth[0],
               time_step_size = time_step_size,
               agent_step_size = agent_step_size)
    envB = MergingRoom2D(environment_name=env_name[1],
               room_width = room_width,
               room_depth = room_depth[1],
               time_step_size = time_step_size,
               agent_step_size = agent_step_size)
    envAB = MergingRoom2D(environment_name=env_name[2],
               room_width = room_width,
               room_depth = room_depth[2],
               time_step_size = time_step_size,
               agent_step_size = agent_step_size)

    envs = [envA, envB, envAB]

    agent = RandomAgent()

    n_steps = 1000

    for env in envs:
        # Initialize environment
        obs, state = env.reset()
        for i in range(n_steps):
            # Observe to choose an action
            action = agent.act(obs)
            # Run environment for given action
            obs, state, reward = env.step(action)

    ax = env.plot_trajectory()
    fontsize = 16
    ax.grid()
    # ax.legend(fontsize=fontsize, loc="upper left")
    ax.set_xlabel("width", fontsize=fontsize)
    ax.set_ylabel("depth", fontsize=fontsize)
    plt.show()