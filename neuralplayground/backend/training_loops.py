from neuralplayground.agents import AgentCore
from neuralplayground.arenas import Environment


def default_training_loop(agent: AgentCore, env: Environment, n_steps: int):
    """Default training loop for agents and environments that use a step-based update.

    Parameters
    ----------
    agent : AgentCore
        Agent to be trained.
    env : Environment
        Environment in which the agent is trained.
    n_steps : int
        Number of steps to train the agent for.

    Returns
    -------
    agent : AgentCore
        Trained agent.
    env : Environment
        Environment in which the agent was trained.
    dict_training : dict
        Dictionary containing the training history from the training loop and update method.
    """

    obs, state = env.reset()
    training_hist = []
    obs = obs[:2]
    for j in range(round(n_steps)):
        # Observe to choose an action
        action = agent.act(obs)
        # Run environment for given action
        obs, state, reward = env.step(action)
        update_output = agent.update()
        training_hist.append(update_output)
        obs = obs[:2]
    process_training_hist(training_hist)
    return agent, env, dict


def episode_based_training_loop(agent: AgentCore, env: Environment, t_episode: int, n_episode: int):
    """Training loop for agents and environments that use an episode-based update.

    Parameters
    ----------
    agent : AgentCore
        Agent to be trained.
    env : Environment
        Environment in which the agent is trained.
    t_episode : int
        Number of steps per episode.
    n_episode : int
        Number of episodes to train the agent for.

    Returns
    -------
    agent : AgentCore
        Trained agent.
    env : Environment
        Environment in which the agent was trained.
    dict_training : dict
        Dictionary containing the training history from the training loop and update method.
    """

    obs, state = env.reset()
    obs = obs[:2]
    training_hist = []
    for i in range(n_episode):
        for j in range(t_episode):
            action = agent.act(obs)
            update_output = agent.update()
            training_hist.append(update_output)
            obs, state, reward = env.step(action)
            obs = obs[:2]
    dict_training = process_training_hist(training_hist)
    return agent, env, dict_training


def tem_training_loop(agent: AgentCore, env: Environment, n_steps: int):
    pass


def process_training_hist(training_hist):
    """Process the training history from the training loop and update method.

    Parameters
    ----------
    training_hist : list
        List of dictionaries containing the training history from the training loop and update method.

    Returns
    -------
    dict_training : dict
        Dictionary containing the one list per key in the training_hist. The list contains the values for
        that key for each step in the training loop.
    """

    dict_training = {}
    for key in training_hist[0].keys():
        dict_training[key] = []
    for i in range(len(training_hist)):
        for key in training_hist[i].keys():
            dict_training[key].append(training_hist[i][key])
    return dict_training
