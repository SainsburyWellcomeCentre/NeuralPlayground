import numpy as np
from .agent_core import AgentCore

class SimpleDiscreteAgent(AgentCore):
    """
    A simplified single-environment discrete agent, loosely mirroring TEM’s
    approach to picking actions and checking whether the environment
    actually moved.
    """

    def __init__(
        self,
        agent_name: str = "SimpleDiscreteAgent",
        **model_kwargs,
    ):
        """
        Parameters
        ----------
        room_width : int
            Width (in discrete states) of the environment
        room_depth : int
            Depth (in discrete states) of the environment
        state_density : float
            Number of discrete states per unit distance (usually 1 / step_size)
        agent_name : str
            Agent's name
        """
        super().__init__(agent_name=agent_name)
        self.room_width = model_kwargs["room_width"]
        self.room_depth = model_kwargs["room_depth"]
        self.state_density = model_kwargs["state_density"]
        # Discrete actions: stay, up, down, right, left
        self.poss_actions = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]

        # For storing trajectory
        self.walk_actions = []
        self.obs_history = []

        # Keep track of previous observation/action so we know if the environment actually moved.
        self.prev_observation = None
        self.prev_action = [0, 0]
        self.n_walk = 0

    def reset(self):
        """
        Reset the agent’s history and counters.
        """
        super().reset()
        self.walk_actions = []
        self.obs_history = []
        self.prev_observation = None
        self.prev_action = [0, 0]
        self.n_walk = 0

    def act(self, observation, policy_func=None):
        """
        Decide on the next action. If the environment did not change state
        (i.e. we got the same position as before, and we tried to move),
        then pick a new random action. Otherwise, record the old observation and action.

        Parameters
        ----------
        observation : list or np.ndarray
            Typically [state_index, object_info, (x,y)] for a discrete environment.
            The first element (observation[0]) is the discrete state index.

        policy_func : callable, optional
            Unused here. Included only for compatibility.

        Returns
        -------
        action : list
            Chosen discrete action [dx, dy]
        """
        # If this is our first time calling act, initialise
        if self.prev_observation is None:
            self.prev_observation = observation
            self.prev_action = self.action_policy()
            return self.prev_action

        # Check if environment actually moved to a new state
        curr_state_idx = observation[0]
        prev_state_idx = self.prev_observation[0]

        if curr_state_idx == prev_state_idx and self.prev_action != [0, 0]:
            # The environment didn't move from last action, so pick a new random action
            new_action = self.action_policy()
        else:
            # The environment did move, so record old obs/action before picking the next action
            self.walk_actions.append(self.prev_action)
            self.obs_history.append(self.prev_observation)
            self.n_walk += 1
            new_action = self.action_policy()

        self.prev_observation = observation
        self.prev_action = new_action
        return new_action

    def action_policy(self):
        """
        Random action policy that selects an action from [stay, up, down, right, left].
        """
        idx = np.random.choice(len(self.poss_actions))
        return self.poss_actions[idx]

    def update(self):
        """
        Update the agent's internal state after a walk is completed.
        """
        self.n_walk = 0
        return None