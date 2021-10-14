import sys
sys.path.append("../")
import matplotlib.pyplot as plt
from environments.environment import Environment
import numpy as np


class Simple2D(Environment):

    def __init__(self, environment_name="2DEnv", **env_kwargs):
        super().__init__(environment_name, **env_kwargs)
        self.metadata = {"env_kwargs": env_kwargs}
        self.room_width, self.room_depth = env_kwargs["room_width"], env_kwargs["room_depth"]
        self.agent_step_size = env_kwargs["agent_step_size"]
        self.reset()

    def reset(self):
        """ Start in a random position within the dimensions of the room """
        self.global_steps = 0
        self.history = []
        self.state = [np.random.uniform(low=-self.room_width/2, high=self.room_width/2),
                      np.random.uniform(low=-self.room_depth/2, high=self.room_depth/2)]
        self.state = np.array(self.state)
        # Fully observable environment, make_observation returns the state
        observation = self.make_observation()
        return observation, self.state

    def step(self, action):
        """ Action should be a vector indicating the direction of the step """
        self.global_steps += 1
        action = action/np.linalg.norm(action)
        new_state = self.state + self.agent_step_size*action
        new_state = np.array([np.clip(new_state[0], a_min=-self.room_width/2, a_max=self.room_width/2),
                              np.clip(new_state[1], a_min=-self.room_depth/2, a_max=self.room_depth/2)])
        reward = 0  # If you get reward, it should be coded here
        transition = {"action": action, "state": self.state, "next_state": new_state,
                      "reward": reward, "step": self.global_steps}
        self.history.append(transition)
        self.state = new_state
        observation = self.make_observation()
        return observation, new_state, reward

    def plot_trajectory(self, history_data=None, ax=None):
        if history_data is None:
            history_data = self.history
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(8, 6))

        ax.plot([-self.room_width/2, self.room_width/2],
                [-self.room_depth/2, -self.room_depth/2], "k", lw=2)
        ax.plot([-self.room_width/2, self.room_width/2],
                [self.room_depth/2, self.room_depth/2], "k", lw=2)
        ax.plot([-self.room_width/2, -self.room_width/2],
                [-self.room_depth/2, self.room_depth/2], "k", lw=2)
        ax.plot([self.room_width / 2, self.room_width / 2],
                [-self.room_depth / 2, self.room_depth / 2], "k", lw=2)

        state_history = [s["state"] for s in history_data]
        next_state_history = [s["next_state"] for s in history_data]
        starting_point = state_history[0]
        ending_point = next_state_history[-1]
        print(starting_point)

        for i, s in enumerate(state_history):
            x_ = [s[0], next_state_history[i][0]]
            y_ = [s[1], next_state_history[i][1]]
            ax.plot(x_, y_, "C0-o", alpha=0.6)

        ax.plot(starting_point[0], starting_point[1], "C3*", ms=13, label="starting point")
        ax.plot(ending_point[0], ending_point[1], "C2*", ms=13, label="ending point")
        return ax
