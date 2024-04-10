import numpy as np

from neuralplayground.replaybuffer import Transition, UniformReplay


class PrioritizedReplay(UniformReplay):
    def __init__(self, buffer_size, prioritize_method="prop", alpha=0.6, beta=0.4):
        """Initialize the replay buffer.
        ----------
        Parameters
        buffer_size : int
            Maximum size of the buffer.
        prioritize_method : str
            Proportional, ranked or greedy as in Prioritized Experience Replay.
        alpha : float
            Alpha parameter for prioritized experience replay.
        beta : float
            Beta parameter for prioritized experience replay (important sampling for bias correction).
        """
        super().__init__(buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.prioritize_method = prioritize_method
        self.transition_priorities_variables = []
        self.transition_prob = None

    def add_transition_obj(self, transition: Transition):
        """Add a transition object to the buffer."""
        # If the buffer is full, pop the oldest transition
        super(PrioritizedReplay, self).add_transition_obj(transition)
        if len(self.buffer) >= self.buffer_size:
            self.transition_priorities_variables.pop(0)
        self.transition_priorities_variables.append(transition.priority)

    def _update_probability(self):
        """Update the probability of sampling each transition."""
        priority_variables = np.array(self.transition_priorities_variables)
        if self.prioritize_method == "rank":
            ranks = np.flip(np.argsort(priority_variables))
            self.transition_prob = 1 / (1 + ranks)
        elif self.prioritize_method == "prop":
            priority = priority_variables + 1e-8
            self.transition_prob = priority / np.sum(priority)
        elif self.prioritize_method == "greedy":
            max_td_index = np.argmax(priority_variables)
            self.transition_prob = np.zeros(
                shape=(
                    len(
                        self.buffer,
                    )
                )
            )
            self.transition_prob[max_td_index] = 1.0
        else:
            raise ValueError("Invalid prioritize method")

    def _update_weights(self):
        """Update the importance weights."""
        if self.prioritize_method == "greedy":
            self.importance_weights = np.ones(
                shape=(
                    len(
                        self.transition_prob,
                    )
                )
            )
        else:
            if self.beta != 0:
                self.importance_weights = (1 / (len(self.transition_prob) * (self.transition_prob + 1e-10))) ** self.beta
                self.importance_weights = self.importance_weights / np.max(self.importance_weights)
            else:
                self.importance_weights = np.ones(self.transition_prob.shape)

    def sample_batch(self, batch_size: int, return_update_weights=False):
        """Sample a batch of transitions."""
        self._update_probability()
        self._update_weights()
        indexes = np.random.choice(len(self.buffer), batch_size, p=self.transition_prob)
        transitions = np.array(self.buffer)[indexes]
        if return_update_weights:
            return transitions, self.importance_weights[indexes]
        else:
            return transitions
