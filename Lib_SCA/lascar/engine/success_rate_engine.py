import numpy as np


class numerical_success_rate:
    """
    this is a numerical evaluation of success rate without employing multiple attacks
    reference: Rivain, Matthieu. "On the exact success rate of side channel analysis in the gaussian model"
    ## parameters ##
    distinguish_vector: attack results from one type of attack (cpa or dpa), shape usually is (number of key guesses * time)
    correct_key: index of correct key guess (priori info)
    order: the order of success rate
    ## main idea ##
    we assume that the distribution of the distinguishing vector d is a multivariate Gaussian,
    and the success rate can be expressed as a sum of Gaussian cumulative distribution functions (cdf).
    """
    def __init__(self, distinguish_vector, correct_key, order=1):
        self.distinguish_vector = distinguish_vector
        self.no_key_guesses = self.distinguish_vector.shape[0]
        self.no_time_samples = self.distinguish_vector.shape[1]
        self.correct_key = correct_key
        self.order = order

    def eval(self):
        # construct comparison vector c - difference of distinguish_vector (d)
        d_correct = self.distinguish_vector[self.correct_key]
        c = np.zeros((self.no_key_guesses - 1, self.no_time_samples))
        ci = 0
        for key_id in range(self.no_time_samples):
            if not key_id == self.correct_key:
                c[ci] = d_correct - self.distinguish_vector[key_id]

        # using maximum likelihood to estimate parameters of multivariate Gaussian distribution

