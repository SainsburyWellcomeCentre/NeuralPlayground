import numpy as np
import matplotlib.pyplot as plt
from neuralplayground.agents import LevyFlightAgent
from neuralplayground.arenas import Simple2D, ConnectedRooms


def main():
    agent = LevyFlightAgent(step_size=0.8, scale=2.0, loc=0.0, beta=1.0, alpha=0.5, max_action_size=100)
    time_step_size = 0.1 #seg
    agent_step_size = 3

    # Init environment
    env = Simple2D(time_step_size = time_step_size,
                   agent_step_size = agent_step_size,
                   arena_x_limits=(-20, 20), 
                   arena_y_limits=(-20, 20))

    n_steps = 5000#50000

    # Initialize environment
    obs, state = env.reset()
    for i in range(n_steps):
        # Observe to choose an action
        action = agent.act(obs)
        # Run environment for given action
        obs, state, reward = env.step(action)
        env.render()


if __name__ == "__main__":
    main()