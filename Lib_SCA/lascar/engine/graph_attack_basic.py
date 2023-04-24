import collections
import gc
from bisect import bisect_left
from math import floor, log
from tqdm import tqdm
import numpy as np
from scipy.stats import norm, bernoulli, chi2, ks_2samp, cramervonmises_2samp, pearsonr, rv_histogram
from scipy.linalg import orthogonal_procrustes, eigvals, eigvalsh
import netcomp as nc
from sklearn.cluster import KMeans
from TracyWidom import TracyWidom
from scipy.io import loadmat

gamma_diff_save = list()
miu_diff_save, rd_save, epsilon_save = [list(), list()], [list(), list()], [list(), list()]


class SpectralDistance:

    def __init__(self, mode='l2', k=3):
        """
        :param k: k largest eigenvalues to be compared
        :param mode: 'l2', 'lmax', 'ks' --> Kolmogorov–Smirnov (K-S) distance/statistic
        """
        self.k = k
        if mode == 'l2':
            self.spectral_distance = self.l2
        elif mode == 'lmax':
            self.spectral_distance = self.lmax
        elif mode == 'ks':
            self.spectral_distance = self.ks
        elif mode == 'edit':
            self.spectral_distance = self.edit_distance

    def l2(self, a, b):
        eigenvalue_a = eigvalsh(a)
        eigenvalue_b = eigvalsh(b)
        eigenvalue_a = eigenvalue_a[::-1]
        eigenvalue_b = eigenvalue_b[::-1]
        dist = np.sqrt(np.sum((eigenvalue_a[:self.k] - eigenvalue_b[:self.k]) ** 2))
        return dist

    def lmax(self, a, b):
        eigenvalue_a = eigvalsh(a)
        eigenvalue_b = eigvalsh(b)
        eigenvalue_a = eigenvalue_a[::-1]
        eigenvalue_b = eigenvalue_b[::-1]
        dist = np.max(np.abs(eigenvalue_a[:self.k] - eigenvalue_b[:self.k]))
        return dist

    def ks(self, a, b):
        eigenvalue_a = eigvalsh(a)
        eigenvalue_b = eigvalsh(b)
        eigenvalue_a = eigenvalue_a[::-1]
        eigenvalue_b = eigenvalue_b[::-1]
        sak, sbk = eigenvalue_a[:self.k], eigenvalue_b[:self.k]
        res = ks_2samp(sak, sbk)
        return res.statistic

    @staticmethod
    def edit_distance(a, b):
        return nc.edit_distance(a, b)
