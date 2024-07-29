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
    dict_training = process_training_hist(training_hist)
    return agent, env, dict_training


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


def tem_training_loop(
    agent: AgentCore, env: Environment, n_episode: int, params: dict, random_state: bool = True, custom_state: list = None
):
    """Training loop for agents and environments that use a TEM-based update.

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
    import numpy as np

    training_dict = [agent.mod_kwargs, env.env_kwargs, agent.tem.hyper]

    max_steps_per_env = np.random.randint(4000, 6000, size=params["batch_size"])
    current_steps = np.zeros(params["batch_size"], dtype=int)

    obs, state = env.reset(random_state=random_state, custom_state=custom_state)

    for i in range(n_episode):
        while agent.n_walk < params["n_rollout"]:
            actions = agent.batch_act(obs)
            obs, state, reward = env.step(actions, normalize_step=True)
        agent.update()
        current_steps += params["n_rollout"]
        finished_walks = current_steps >= max_steps_per_env

        if any(finished_walks):
            for env_i in np.where(finished_walks)[0]:
                env.reset_env(env_i)
                agent.prev_iter[0].a[env_i] = None

                max_steps_per_env[env_i] = np.random.randint(4000, 6000)
                current_steps[env_i] = 0
    return agent, env, training_dict


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
    if training_hist[0] is None:
        dict_training = None
    else:
        for key in training_hist[0].keys():
            dict_training[key] = []
        for i in range(len(training_hist)):
            for key in training_hist[i].keys():
                dict_training[key].append(training_hist[i][key])
    return dict_training
