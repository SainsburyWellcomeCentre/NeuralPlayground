import numpy as np
from .agent_core import AgentCore

class SimpleDiscreteAgent(AgentCore):
    """
    A simplified single-environment discrete agent, with optional square or hex action sets.
    """

    def __init__(
        self,
        agent_name: str = "SimpleDiscreteAgent",
        **model_kwargs,
    ):
        """
        Parameters
        ----------
        grid_type : {'square', 'hex'}
            Determines the action set. 'square' uses [stay, up, down, left, right].
            'hex' uses [stay] plus six hexagonal directions.
        agent_name : str
            Agent's name
        """
        super().__init__(agent_name=agent_name)
        self.room_width = model_kwargs["room_width"]
        self.room_depth = model_kwargs["room_depth"]
        self.state_density = model_kwargs["state_density"]
        self.grid_type = model_kwargs["grid_type"]

        # Configure possible actions
        if self.grid_type == "square":
            # Discrete actions: stay, up, down, right, left
            self.poss_actions = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]
        else:
            # For hex, we can define six directions in 2D plus stay.
            # Below is one common set for a pointy-topped hex layout,
            # but you can adjust as desired.
            sqrt3_2 = np.sqrt(3) / 2
            self.poss_actions = [
                [0, 0],                 # stay still
                [1.0, 0.0],            # east
                [0.5,  sqrt3_2],       # northeast
                [-0.5,  sqrt3_2],      # northwest
                [-1.0, 0.0],           # west
                [-0.5, -sqrt3_2],      # southwest
                [0.5, -sqrt3_2],       # southeast
            ]

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
        if self.prev_observation is None:
            # First time: just pick an action and store current observation
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
            # The environment did move, so record old obs/action before picking the next
            self.walk_actions.append(self.prev_action)
            self.obs_history.append(self.prev_observation)
            self.n_walk += 1
            new_action = self.action_policy()

        self.prev_observation = observation
        self.prev_action = new_action
        return new_action

    def action_policy(self):
        """
        Random action policy that selects among the pre-defined action set.
        """
        idx = np.random.choice(len(self.poss_actions))
        return self.poss_actions[idx]

    def update(self):
        """
        No-op update in this simplified agent.
        """
        self.n_walk = 0
        return None