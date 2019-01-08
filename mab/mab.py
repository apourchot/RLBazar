import numpy as np

class UCB:

    def __init__(self, n_arms, args):

        # bandit stuff
        self.n_arms = n_arms
        self.confidence_bounds = np.zeros(n_arms)

        # misc
        self.args = args

    def sample(self):
        """
        Samples an arm
        """
        return i

    def update_arm(self, i, score):
        """
        Updates the averages and confidence bounds
        """
