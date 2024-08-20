from neuralplayground.agents import RandomAgent, LevyFlightAgent,TrajectoryGenerator
# Random agent generates a brownian motion. Levy flight is still experimental.

agent = TrajectoryGenerator()
time_step_size = 0.1 #seg
agent_step_size = 3

# Init environment
env = Simple2D(time_step_size = time_step_size,
               agent_step_size = agent_step_size,
               arena_x_limits=(-100, 100),
               arena_y_limits=(-100, 100))


print('testing stuff ')