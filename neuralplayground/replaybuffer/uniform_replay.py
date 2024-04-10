import numpy as np

from neuralplayground.replaybuffer import Transition


class UniformReplay(object):
    """Uniform replay buffer, samples transitions uniformly.
    Once it reaches the buffer size, it will start to replace the oldest transitions as a FIFO buffer.
    ----------
    Attributes
    buffer_size : int
        Maximum size of the buffer.
    buffer : list
        List of transitions.
    ----------
    Methods
    add_transition_obj(transition: Transition)
        Add a transition object to the buffer.
    add_transition(prev_state, action, reward, next_state, terminate, priority=None)
        Add a transition to the buffer.
    sample(batch_size)
        Sample a batch of transitions.
    get_last_transition()
        Return the last transition in the buffer.
    set_alpha(alpha)
        Set the alpha parameter for prioritized experience replay.
    set_beta(beta)
        Set the beta parameter for prioritized experience replay.
    """

    def __init__(self, buffer_size):
        """Initialize the replay buffer."""
        self.buffer_size = buffer_size
        self.buffer = []

    def add_transition_obj(self, transition: Transition):
        """Add a transition object to the buffer."""
        # If the buffer is full, pop the oldest transition
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def add_transition(self, prev_state, action, reward, next_state, terminate, priority=None):
        """Add a transition to the buffer."""
        transition = Transition(prev_state, action, reward, next_state, terminate, priority)
        self.add_transition_obj(transition)

    def sample_batch(self, batch_size: int):
        """Sample a batch of transitions."""
        indexes = np.random.choice(len(self.buffer), batch_size, replace=False)
        transitions = np.array(self.buffer)[indexes]
        return transitions

    def get_last_transition(self):
        """Return the last transition in the buffer."""
        return self.buffer[-1]

    def __len__(self):
        return len(self.buffer)

    def set_alpha(self, alpha: float):
        pass

    def set_beta(self, beta: float):
        pass
