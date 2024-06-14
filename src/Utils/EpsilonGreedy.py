import random
import numpy as np
random.seed(0)


class EpsilonGreedy():
    def __init__(self, eps_start=1, eps_end=0.001, eps_decay=0.999):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def act(self, q_values):
        """
        Choose an action based on the epsilon-greedy strategy.

        Arguments:
        - q_values: A list of Q-values representing the expected rewards for each action.

        Returns:
        - The chosen action index.
        """
        if random.random() < self.eps_start:
            return random.choice(range(len(q_values)))  # Randomly choose an action index
        else:
            return np.argmax(q_values)  # Choose the action with the highest Q-value

    def update_epsilon(self):
        """
        Update the epsilon value based on the decay rate.
        """
        self.eps_start = max(self.eps_end, self.eps_start * self.eps_decay)  # Update the epsilon value using the decay rate