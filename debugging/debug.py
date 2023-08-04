from neuralplayground.agents import LevyFlightAgent
from neuralplayground.arenas import Environment

n_steps = 5000  # 50000
# Random agent generates a brownian motion. Levy flight is still experimental.
agent = LevyFlightAgent(step_size=0.8, scale=2.0, loc=0.0, beta=1.0, alpha=0.5, max_action_size=100)
time_step_size = 0.1  #
env = Environment(environment_name="test_2", time_step_size=time_step_size)

# Initialize environment
obs, state = env.reset()
for i in range(n_steps):
    # Observe to choose an action
    action = agent.act(obs)
    # Run environment for given action
    obs, state, reward = env.step(action)
print(env.global_time)
print(env)
print(env.global_time)
env.restore_environment("./saved")
print(env)
print(env.global_time)
