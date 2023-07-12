from neuralplayground.agents import AgentCore
from neuralplayground.arenas import Environment


def default_training_loop(agent: AgentCore, env: Environment, n_steps: int):
    obs, state = env.reset()
    for j in range(round(n_steps)):
        # Observe to choose an action
        action = agent.act(obs)
        # Run environment for given action
        obs, state, reward = env.step(action)
        agent.update()
    return agent, env


def episode_based_training_loop(agent: AgentCore, env: Environment, t_episode: int, n_episode: int):
    obs, state = env.reset()
    obs = obs[:2]
    for i in range(n_episode):
        for j in range(t_episode):
            action = agent.act(obs)
            agent.update()
            obs, state, reward = env.step(action)
            obs = obs[:2]

    return agent, env


def merge_room_training(agent: AgentCore, env: Environment, n_steps: int):
    pass


def tem_training_loop(agent: AgentCore, env: Environment, n_steps: int):
    pass
