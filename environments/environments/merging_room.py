import sys
sys.path.append("../")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from simple2d import Simple2D

class MergingRoom2D(Simple2D):
    def __init__(self, environment_name="MergingRoom", **env_kwargs):
        self.environment_name = environment_name
        env_kwargs["room_width"] = 200
        env_kwargs["room_depth"] = 200
        self.merge_time = 10
        self.switch_time = 5
        self.run_full_experiment = True
        super().__init__(environment_name, **env_kwargs)
        self.AB_limits = np.array([[0, self.room_width], [-self.room_depth/2, self.room_depth/2]])
        self.AB_id = "AB"
        self.A_limits = np.array([[0, self.room_width], [0, self.room_depth/2]])
        self.A_id = "A"
        self.B_limits = np.array([[0, self.room_width], [-self.room_depth/2, 0]])
        self.B_id = "B"

    def set_room(self, room_id):
        if room_id == self.A_id:
            self.arena_limits = self.A_limits
        elif room_id == self.B_id:
            self.arena_limits = self.B_limits
        elif room_id == self.AB_id:
            self.arena_limits = self.AB_limits
        
        return self.arena_limits

    def step(self, action, room_id):
        if self.run_full_experiment == True:
            self.arena_limits = self.set_room(room_id)
            # if self.global_steps == 0:
            #     self.arena_limits = self.set_room("A")
            # elif self.global_steps == 333:
            #     self.arena_limits = self.set_room("B")
            # elif self.global_steps == 666:
            #     self.arena_limits = self.set_room("AB")


        self.global_steps += 1
        action = action/np.linalg.norm(action)
        new_state = self.state + self.agent_step_size*action
        new_state = np.array([np.clip(new_state[0], a_min=self.arena_limits[0,0], a_max=self.arena_limits[0,1]),
                              np.clip(new_state[1], a_min=self.arena_limits[1,0], a_max=self.arena_limits[1,1])])
        reward = 0  # If you get reward, it should be coded here
        transition = {"action": action, "state": self.state, "next_state": new_state,
                      "reward": reward, "step": self.global_steps}
        self.history.append(transition)
        self.state = new_state
        observation = self.make_observation()
        
        return observation, new_state, reward

    def plot_trajectory(self, room, history_data=None, ax=None):
        """ Plot the Trajectory of the agent in the environment

        Parameters
        ----------
        history_data: None
            default to access to the saved history of positions in the environment
        ax: None
            default to create ax

        Returns
        -------
        Returns a plot of the trajectory of the animal in the environment
        """
        if history_data is None:
            history_data = self.history
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(8, 6))

        ax.plot([0, self.room_width],
                [-self.room_depth/2, -self.room_depth/2], "r", lw=2)
        ax.plot([0, self.room_width],
                [self.room_depth/2, self.room_depth/2], "r", lw=2)
        ax.plot([self.room_width, self.room_width],
                [-self.room_depth/2, self.room_depth/2], "r", lw=2)
        ax.plot([0, 0],
                [-self.room_depth / 2, self.room_depth / 2], "r", lw=2)
        
        if room == "B":
            ax.plot([0, self.room_width],
                [0, 0], "r", lw=2)

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

class RandomAgent(object):
    
    def act(self, observation):
        return np.random.normal(scale=0.1, size=(2,))

if __name__ == "__main__":
    env_name = "MergingRoom"
    rooms = ["A", "B", "AB"]
    time_step_size = 0.1
    agent_step_size = 5
    n_steps = 1000

    env = MergingRoom2D(environment_name=env_name,
                        time_step_size=time_step_size,
                        agent_step_size=agent_step_size)
    agent = RandomAgent()

    # Initialize environment
    obs, state = env.reset()
    for room in rooms:
        for j in range(n_steps):
            # Observe to choose an action
            action = agent.act(obs)
            # Run environment for given action
            obs, state, reward = env.step(action, room)

        if room == "B":
            ax1 = env.plot_trajectory(room)
            fontsize = 16
            ax1.grid()
            # ax.legend(fontsize=fontsize, loc="upper left")
            ax1.set_xlabel("width", fontsize=fontsize)
            ax1.set_ylabel("depth", fontsize=fontsize)

            # Initialize environment
            obs, state = env.reset()
        
        if room == "AB":
            ax2 = env.plot_trajectory(room)
            fontsize = 16
            ax2.grid()
            # ax.legend(fontsize=fontsize, loc="upper left")
            ax2.set_xlabel("width", fontsize=fontsize)
            ax2.set_ylabel("depth", fontsize=fontsize)
    
    
    plt.show()