import sys
sys.path.append("../")
import matplotlib.pyplot as plt
import numpy as np

class RandomAgent(object):
    
    def act(self, observation):
        return np.random.normal(scale=0.1, size=(2,))

if __name__ == "__main__":
    from environments.core import Environment
    from simple2d import Simple2D

    class MergingRoom2D(Simple2D):
        def __init__(self, environment_name="MergingRoom", **env_kwargs):
            self.environment_name = environment_name
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
            if room_id == self.AB_id:
                self.arena_limits = self.AB_limits

            if room_id == self.A_id:
                self.arena_limits = self.A_limits
            
            if room_id == self.B_id:
                self.arena_limits = self.B_limits

            return self.arena_limits

        def step(self, action):
            if self.run_full_experiment == True:
                if self.global_steps == 0:
                    self.arena_limits = self.set_room(self, "A")
                elif self.global_steps == 5:
                    self.arena_limits = self.set_room(self, "B")
                elif self.global_steps == 10:
                    self.arena_limits = self.set_room(self, "AB")

            return super().step(action)

    room_width = 200
    room_depth = [100, 100, 200]
    time_step_size = 0.1
    agent_step_size = 0.5
    n_steps = 1000

    env = MergingRoom2D(environment_name="MergingRoom")
    agent = RandomAgent()

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