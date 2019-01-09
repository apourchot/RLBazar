import numpy as np


class UCB:
    """
    Upper Confidence Bound algorithm
    """

    def __init__(self, n_arms, args=None):

        # bandit stuff
        self.n_arms = n_arms
        self.means = np.zeros(n_arms)
        self.occurences = np.ones(n_arms)
        self.f = lambda t: 1 + t * np.log(t) ** 2
        self.t = 0

        # misc
        self.args = args
        self.alpha = 0.9

    def sample(self):
        """
        Samples an arm
        """

        # if we tried all arms at least once
        if self.t >= self.n_arms:
            confidence_bounds = self.compute_confidences()
            i = np.argmax(confidence_bounds)
        else:
            i = self.t
        self.t += 1

        return i

    def compute_confidences(self):
        """
        Computes confidence bounds
        """
        return self.means + 100 * np.sqrt(2 * np.log(self.f(self.t)) / self.occurences)

    def update_arm(self, i, score):
        """
        Updates the averages and confidence bounds
        """
        if self.occurences[i] > 1:
            self.means[i] = self.alpha * self.means[i] + (1 - self.alpha) * score
        else:
            self.means[i] = score
        self.occurences[i] += 1
