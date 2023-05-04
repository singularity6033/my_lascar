import numpy as np
from scipy.stats import ks_2samp

gamma_diff_save = list()
miu_diff_save, rd_save, epsilon_save = [list(), list()], [list(), list()], [list(), list()]


class GraphDistance:

    def __init__(self, mode='sd_l2', k=10):
        self.k = k

        if mode == 'sd_l2':
            self.graph_distance = self.sd_l2
        elif mode == 'sd_lmax':
            self.graph_distance = self.sd_lmax
        elif mode == 'sd_ks':
            self.graph_distance = self.sd_ks
        elif mode == 'edit':
            self.graph_distance = self.edit_distance

    def sd_l2(self, a, b):
        eigenvalue_a = np.linalg.eigvalsh(a)
        eigenvalue_b = np.linalg.eigvalsh(b)
        eigenvalue_a = eigenvalue_a[::-1]
        eigenvalue_b = eigenvalue_b[::-1]
        dist = np.sqrt(np.sum((eigenvalue_a[:self.k] - eigenvalue_b[:self.k]) ** 2))
        return dist

    def sd_lmax(self, a, b):
        eigenvalue_a = np.linalg.eigvalsh(a)
        eigenvalue_b = np.linalg.eigvalsh(b)
        eigenvalue_a = eigenvalue_a[::-1]
        eigenvalue_b = eigenvalue_b[::-1]
        dist = np.max(np.abs(eigenvalue_a[:self.k] - eigenvalue_b[:self.k]))
        return dist

    def sd_ks(self, a, b):
        # Kolmogorov–Smirnov(K - S) distance / statistic
        eigenvalue_a = np.linalg.eigvalsh(a)
        eigenvalue_b = np.linalg.eigvalsh(b)
        eigenvalue_a = eigenvalue_a[::-1]
        eigenvalue_b = eigenvalue_b[::-1]
        sak, sbk = eigenvalue_a[:self.k], eigenvalue_b[:self.k]
        res = ks_2samp(sak, sbk)
        return res.statistic

    @staticmethod
    def edit_distance(a, b):
        dist = np.abs((a - b)).sum() / 2
        return dist
