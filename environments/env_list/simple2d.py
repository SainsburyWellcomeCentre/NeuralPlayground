import sys
sys.path.append("../")
import matplotlib as mpl
import matplotlib.pyplot as plt
from environments.environment import Environment
import numpy as np
from environments.experiment_data.behavioral_data import SargoliniData, FullSargoliniData,FullHaftingData



class Simple2D(Environment):
    def __init__(self, environment_name="2DEnv", **env_kwargs):
        super().__init__(environment_name, **env_kwargs)
        self.metadata = {"env_kwargs": env_kwargs}
        self.room_width, self.room_depth = env_kwargs["room_width"], env_kwargs["room_depth"]
        self.arena_limits = np.array([[-self.room_width/2, self.room_width/2],
                                      [-self.room_depth/2, self.room_depth/2]])
        self.agent_step_size = env_kwargs["agent_step_size"]
        self.state_dims_labels = ["x_pos", "y_pos"]
        self.reset()

    def reset(self):
        """ Start in a random position within the dimensions of the room """
        self.global_steps = 0
        self.history = []
        self.state = [0,0]
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
                [-self.room_depth/2, -self.room_depth/2], "r", lw=2)
        ax.plot([-self.room_width/2, self.room_width/2],
                [self.room_depth/2, self.room_depth/2], "r", lw=2)
        ax.plot([-self.room_width/2, -self.room_width/2],
                [-self.room_depth/2, self.room_depth/2], "r", lw=2)
        ax.plot([self.room_width / 2, self.room_width / 2],
                [-self.room_depth / 2, self.room_depth / 2], "r", lw=2)

        state_history = [s["state"] for s in history_data]
        next_state_history = [s["next_state"] for s in history_data]
        starting_point = state_history[0]
        ending_point = next_state_history[-1]
        print(starting_point)

        cmap = mpl.cm.get_cmap("plasma")
        norm = plt.Normalize(0, len(state_history))

        aux_x = []
        aux_y = []
        for i, s in enumerate(state_history):
            x_ = [s[0], next_state_history[i][0]]
            y_ = [s[1], next_state_history[i][1]]
            aux_x.append(s[0])
            aux_y.append(s[1])
            ax.plot(x_, y_, "-", color=cmap(norm(i)), alpha=0.6)

        sc = ax.scatter(aux_x, aux_y, c=np.arange(len(state_history)),
                        vmin=0, vmax=len(state_history), cmap="plasma", alpha=0.6)

        cbar = plt.colorbar(sc, ax=ax)
        cbar.ax.set_ylabel('N steps', rotation=270)
        return ax


class BasicSargolini2006(Simple2D):

    def __init__(self, data_path="sargolini2006/", environment_name="Sargolini2006", **env_kwargs):
        self.data_path = data_path
        self.environment_name = environment_name
        self.data = SargoliniData(data_path=self.data_path, experiment_name=self.environment_name)
        self.arena_limits = self.data.arena_limits
        self.room_width, self.room_depth = np.abs(np.diff(self.arena_limits, axis=1))
        env_kwargs["room_width"] = self.room_width
        env_kwargs["room_depth"] = self.room_depth
        env_kwargs["agent_step_size"] = 1/50  # In seconds
        super().__init__(environment_name, **env_kwargs)
        self.metadata["doi"] = "https://doi.org/10.1126/science.1125572"

        self.state_dims_labels = ["x_pos", "y_pos", "head_direction_x", "head_direction_y"]

    def reset(self):
        """ Start in a random position within the dimensions of the room """
        self.global_steps = 0
        self.global_time = 0
        self.history = []
        self.pos, self.head_dir = self.data.position[0, :], self.data.head_direction[0, :]
        self.state = np.concatenate([self.pos, self.head_dir])
        # Fully observable environment, make_observation returns the state
        observation = self.make_observation()
        return observation, self.state

    def step(self, action):
        """ Action is ignored in this case """
        self.global_steps += 1
        self.global_time = self.global_steps*self.agent_step_size
        reward = 0  # If you get reward, it should be coded here
        new_state = self.data.position[self.global_steps, :], self.data.head_direction[self.global_steps, :]
        new_state = np.concatenate(new_state)
        transition = {"action": action, "state": self.state, "next_state": new_state,
                      "reward": reward, "step": self.global_steps}
        self.history.append(transition)
        self.state = new_state
        observation = self.make_observation()
        return observation, new_state, reward


class Sargolini2006(Simple2D):

    def __init__(self, data_path="sargolini2006/", environment_name="Sargolini2006", session=None, verbose=False, **env_kwargs):
        self.data_path = data_path
        self.environment_name = environment_name
        self.session = session
        self.data = FullSargoliniData(data_path=self.data_path, experiment_name=self.environment_name, verbose=verbose, session=session)
        self.arena_limits = self.data.arena_limits
        self.room_width, self.room_depth = np.abs(np.diff(self.arena_limits, axis=1))
        self.room_width = self.room_width[0]
        self.room_depth = self.room_depth[0]
        env_kwargs["room_width"] = self.room_width
        env_kwargs["room_depth"] = self.room_depth
        env_kwargs["agent_step_size"] = 1/50  # In seconds
        super().__init__(environment_name, **env_kwargs)
        self.metadata["doi"] = "https://doi.org/10.1126/science.1125572"

        self.state_dims_labels = ["x_pos", "y_pos", "head_direction_x", "head_direction_y"]

    def reset(self):
        """ Start in a random position within the dimensions of the room """
        self.global_steps = 0
        self.global_time = 0
        self.history = []
        self.pos, self.head_dir = self.data.position[0, :], self.data.head_direction[0, :]
        self.state = np.concatenate([self.pos, self.head_dir])
        # Fully observable environment, make_observation returns the state
        observation = self.make_observation()
        return observation, self.state

    def step(self, action):
        """ Action is ignored in this case """
        self.global_steps += 1
        if self.global_steps >= self.data.position.shape[0]-1:
            self.global_steps = 0
        self.global_time = self.global_steps*self.agent_step_size
        reward = 0  # If you get reward, it should be coded here
        new_state = self.data.position[self.global_steps, :], self.data.head_direction[self.global_steps, :]
        new_state = np.concatenate(new_state)
        transition = {"action": action, "state": self.state, "next_state": new_state,
                      "reward": reward, "step": self.global_steps}
        self.history.append(transition)
        self.state = new_state
        observation = self.make_observation()
        return observation, new_state, reward


class Hafting2008(Simple2D):

    def __init__(self, data_path="Hafting2008/C43035A4-5CC5-44F2-B207-126922523FD9_1/", environment_name="Hafting2008", session=None, verbose=False, **env_kwargs):
        self.data_path = data_path
        self.environment_name = environment_name
        self.session = session
        self.data = FullHaftingData(data_path=self.data_path, experiment_name=self.environment_name, verbose=verbose)
        self.arena_limits = self.data.arena_limits
        self.room_width, self.room_depth = np.abs(np.diff(self.arena_limits, axis=1))
        env_kwargs["room_width"] = self.room_width
        env_kwargs["room_depth"] = self.room_depth
        env_kwargs["agent_step_size"] = 1/50  # In seconds
        super().__init__(environment_name, **env_kwargs)
        self.metadata["doi"] = "https://doi.org/10.1038/nature06957"
        self.state_dims_labels = ["x_pos", "y_pos", "head_direction_x", "head_direction_y"]

    def reset(self):
        """ Start in a random position within the dimensions of the room """
        self.global_steps = 0
        self.global_time = 0
        self.history = []
        self.pos, self.head_dir = self.data.position[0, :], self.data.head_direction[0, :]
        self.state = np.concatenate([self.pos, self.head_dir])
        # Fully observable environment, make_observation returns the state
        observation = self.make_observation()
        return observation, self.state

    def step(self, action):
        """ Action is ignored in this case """
        self.global_steps += 1
        self.global_time = self.global_steps*self.agent_step_size
        reward = 0  # If you get reward, it should be coded here
        new_state = self.data.position[self.global_steps, :], self.data.head_direction[self.global_steps, :]
        new_state = np.concatenate(new_state)
        transition = {"action": action, "state": self.state, "next_state": new_state,
                      "reward": reward, "step": self.global_steps}
        self.history.append(transition)
        self.state = new_state
        observation = self.make_observation()
        return observation, new_state, reward


if __name__ == "__main__":
    data_path = "/home/rodrigo/HDisk/8F6BE356-3277-475C-87B1-C7A977632DA7_1/all_data/"
    env = Sargolini2006(data_path=data_path,
                        time_step_size=None,
                        agent_step_size=None)
    env.step()

