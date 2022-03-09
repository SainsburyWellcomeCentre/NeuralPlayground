from re import T
import sys
sys.path.append("../")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from simple2d import Simple2D


class MergingRoom2D(Simple2D):
    def __init__(self, environment_name="MergingRoom", merge_time=100, switch_time=50, **env_kwargs):
        self.environment_name = environment_name
        env_kwargs["room_width"] = 200
        env_kwargs["room_depth"] = 200
        self.time_step_size = time_step_size
        self.merge_time = (merge_time*60) / self.time_step_size
        self.switch_time = (switch_time*60) / self.time_step_size
        self.run_full_experiment = True
        super().__init__(environment_name, **env_kwargs)
        self.AB_limits = np.array([[-self.room_width/2, self.room_width/2], [-self.room_depth / 2, self.room_depth / 2]])
        self.AB_id = "AB"
        self.A_limits = np.array([[-self.room_width/2, self.room_width/2], [-self.room_depth / 2, 0]])
        self.A_id = "A"
        self.B_limits = np.array([[-self.room_width/2, self.room_width/2], [0, self.room_depth / 2]])
        self.B_id = "B"
        self.agent_arena_limits = self.A_limits

    def set_room(self, room_id):
        if room_id == self.A_id:
            self.agent_arena_limits = self.A_limits
        elif room_id == self.B_id:
            self.agent_arena_limits = self.B_limits
        elif room_id == self.AB_id:
            self.agent_arena_limits = self.AB_limits

    def step(self, action):
        if self.run_full_experiment:
            if self.global_steps >= self.merge_time:
                self.set_room("AB")
            elif self.global_steps >= self.switch_time:
                self.set_room("B")

        self.global_steps += 1
        action = action / np.linalg.norm(action)
        new_state = self.state + self.agent_step_size * action
        new_state = np.array([np.clip(new_state[0], a_min=self.agent_arena_limits[0, 0], a_max=self.agent_arena_limits[0, 1]),
                              np.clip(new_state[1], a_min=self.agent_arena_limits[1, 0], a_max=self.agent_arena_limits[1, 1])])
        reward = 0  # If you get reward, it should be coded here
        transition = {"action": action, "state": self.state, "next_state": new_state,
                      "reward": reward, "step": self.global_steps}
        self.history.append(transition)
        self.state = new_state
        observation = self.make_observation()

        return observation, new_state, reward

    def plot_trajectory(self, history_data=None, ax=None):
        if self.run_full_experiment:
            ax = super().plot_trajectory(history_data=history_data)
            if self.global_steps <= self.merge_time:
                ax.plot([-env.room_width/2, env.room_width/2],
                        [0, 0], "r", lw=2)
            else:
                ax.plot([-env.room_width/2, env.room_width/2],
                        [0, 0], "r--", lw=2)
            return ax
        else:
            return super().plot_trajectory(history_data=self.history)


class RandomAgent(object):
    def act(self, observation):
        return np.random.normal(scale=0.1, size=(2,))


if __name__ == "__main__":
    env_name = "MergingRoom"
    time_step_size = 0.2
    agent_step_size = 3
    merging_time = 40
    switch_time = 20
    n_steps = ((merging_time + switch_time)*60) / time_step_size

    env = MergingRoom2D(environment_name=env_name,
                        merge_time=merging_time,
                        switch_time=switch_time,
                        time_step_size=time_step_size,
                        agent_step_size=agent_step_size)
    agent = RandomAgent()

    obs, state = env.reset()
    for j in range(round(n_steps)):
        # Observe to choose an action
        action = agent.act(obs)
        # Run environment for given action
        obs, state, reward = env.step(action)
        if j == ((merging_time*60)/time_step_size)-1:
            ax = env.plot_trajectory()
    merged_history = env.history[int((merging_time*60)/time_step_size):]
    env.plot_trajectory(history_data=merged_history)
    plt.show()