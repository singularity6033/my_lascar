"""
GraphTestEngine is a two-samples graph testing used to compute whether two graph samples G and H are drawn from
the same distribution or not
"""
import collections
import gc
from math import floor, log
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import numpy as np
from scipy.stats import norm, bernoulli, chi2, ks_2samp, cramervonmises_2samp, pearsonr, rv_histogram
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import KMeans
from TracyWidom import TracyWidom
from scipy.io import loadmat

from . import GuessEngine
from .graph_construction_basic import BaseGraph, PhaseSpaceReconstructionGraph, AmplitudesBasedGraph


### one trace converted into one graph ###
class A_GraphConstructionPSR(GuessEngine):
    """
    attack version of PSR graph --> see details in PhaseSpaceReconstructionGraph
    """

    def __init__(self,
                 name,
                 selection_function,
                 guess_range,
                 time_delay=2,
                 dim=3,
                 sampling_interval=1,
                 type='corr',
                 optimal=False,
                 to_unweighted=False,
                 solution=-1
                 ):
        """
        :param selection_function: takes a value and a guess_guess as input, returns 0 or 1
        :param guess_range: what are the values for the guess guess
        :param time_delay: delayed time interval used in phase space reconstruction
        :param dim: the dimension of embedding delayed time series (vectors)
        :param sampling_interval: used to sample the delayed time series, default is 1
        :param type: different measurements to calculate connectivity among nodes (embedded vectors); 'l2' -> 2-norm,
                     'corr' -> pearson correlation coefficient
        :param optimal: False -> use pre-defined delayed time step and embedded dimension in PSRG; True -> use c-c method
                        and fnn to determine the best value for delayed time step and embedded dimension automatically
        :param to_unweighted: save thresholds or not
        :param solution: if known, indicate the correct guess guess.
        """
        self.psr_g = PhaseSpaceReconstructionGraph(time_delay, dim, sampling_interval, type, optimal, to_unweighted)

        self.solution = solution
        GuessEngine.__init__(self, name, selection_function, guess_range, solution)
        self.logger.debug(
            'Creating A_GraphConstructionPSR "%s" with %d guesses.', name, len(guess_range)
        )

    def _initialize(self):
        self.adj_matrix = [defaultdict(list) for _ in range(self._number_of_guesses)]
        self.w_to_unw_thresholds = [defaultdict(list) for _ in range(self._number_of_guesses)]
        self._count = np.zeros((self._number_of_guesses, 2,), np.double)

    def _update(self, batch):
        self._samples_by_selection = [collections.defaultdict(list) for _ in range(self._number_of_guesses)]
        for i in range(len(batch)):
            y = np.array(
                [self._function(batch.values[i], guess) for guess in self._guess_range]
            )

            idx_0 = np.where(y == 0)[0]
            idx_1 = np.where(y == 1)[0]
            g, threshold = self.psr_g.generate(batch.leakages[i])
            for idx_0_i in idx_0:
                self.adj_matrix[idx_0_i][0].append(g)
                if threshold:
                    self.w_to_unw_thresholds[idx_0_i][0].append(threshold)
            self._count[idx_0, 0] += 1

            for idx_1_i in idx_1:
                self.adj_matrix[idx_1_i][1].append(g)
                if threshold:
                    self.w_to_unw_thresholds[idx_1_i][1].append(threshold)
            self._count[idx_1, 1] += 1


class A_GraphConstructionAB(GuessEngine):
    """
    attack version of AB graph --> see details in AmplitudesBasedGraph
    """

    def __init__(self,
                 name,
                 selection_function,
                 guess_range,
                 num_of_amp_groups=10,
                 num_of_moments=4,
                 type='corr',
                 to_unweighted=False,
                 solution=-1
                 ):
        """
        :param selection_function: takes a value and a guess_guess as input, returns 0 or 1
        :param guess_range: what are the values for the guess guess
        :param num_of_amp_groups: number of divided amplitude groups (equally divided between min and max)
        :param num_of_moments: number of the first moments used to construct moments vector
        :param type: different measurements to calculate connectivity among nodes (embedded vectors); 'l2' -> 2-norm,
                     'corr' -> pearson correlation coefficient
        :param solution: if known, indicate the correct guess guess.
        """
        self.num_of_amp_groups = num_of_amp_groups
        self.num_of_moments = num_of_moments
        self.type = type

        self.ab_g = AmplitudesBasedGraph(num_of_amp_groups, num_of_moments, type, to_unweighted)

        self.solution = solution
        GuessEngine.__init__(self, name, selection_function, guess_range, solution)
        self.logger.debug(
            'Creating A_GraphConstructionAB "%s" with %d guesses.', name, len(guess_range)
        )

    def _initialize(self):
        self.adj_matrix = [defaultdict(list) for _ in range(self._number_of_guesses)]
        self.w_to_unw_thresholds = [defaultdict(list) for _ in range(self._number_of_guesses)]
        self._count = np.zeros((self._number_of_guesses, 2,), np.double)
        self.number_of_time_points = self._session.leakage_shape[0]

    def _update(self, batch):
        for i in range(len(batch)):
            y = np.array(
                [self._function(batch.values[i], guess) for guess in self._guess_range]
            )

            idx_0 = np.where(y == 0)[0]
            idx_1 = np.where(y == 1)[0]
            g, threshold = self.ab_g.generate(batch.leakages[i])
            for idx_0_i in idx_0:
                self.adj_matrix[idx_0_i][0].append(g)
                if threshold:
                    self.w_to_unw_thresholds[idx_0_i][0].append(threshold)
            self._count[idx_0, 0] += 1

            for idx_1_i in idx_1:
                self.adj_matrix[idx_1_i][1].append(g)
                if threshold:
                    self.w_to_unw_thresholds[idx_1_i][1].append(threshold)
            self._count[idx_1, 1] += 1


### multiple traces converted into one graph ###
# TODO: this idea is too complex and unreasonable to be used in attack
class A_GraphConstructionTraceBatch(GuessEngine):

    def __init__(self, name, selection_function, guess_range, solution=-1):
        """
        :param selection_function: takes a value and a guess_guess as input, returns 0 or 1
        :param guess_range: what are the values for the guess guess
        :param solution: if known, indicate the correct guess guess.
        """
        self.BaseGraph = BaseGraph()

        self.solution = solution
        GuessEngine.__init__(self, name, selection_function, guess_range, solution)
        self.logger.debug(
            'Creating A_GraphConstructionTraceBatch "%s" with %d guesses.', name, len(guess_range)
        )

    def _initialize(self):
        self.number_of_time_samples = self._session.leakage_shape[0]
        self.embeddings = [None] * (self._number_of_guesses + 1)
        self.results = np.zeros((self._number_of_guesses,) + self._session.leakage_shape, np.double)

    def _update(self, batch):
        lk = batch.leakages
        self.batch_size = lk.shape[0]
        lk_f = np.array(lk.flatten(order='F'), ndmin=2)
        if not isinstance(self.embeddings[0], np.ndarray):
            self.embeddings[0] = lk_f
        else:
            self.embeddings[0] = np.concatenate((self.embeddings[0], lk_f), axis=0)
        y = np.array([self._function(batch.values, guess) for guess in self._guess_range])
        for guess_i in range(len(self._guess_range)):
            yi = np.array(y[guess_i, :], ndmin=2)
            if not isinstance(self.embeddings[guess_i + 1], np.ndarray):
                self.embeddings[guess_i + 1] = yi
            else:
                self.embeddings[guess_i + 1] = np.concatenate((self.embeddings[guess_i+1], yi), axis=0)


### all traces converted into one graph ###
class A_GraphConstructionTraceAllCorr(GuessEngine):
    """
    dpa type attack
    convert all traces in to one graph, each column (along trace axis) represents one node
    using pearson correlation coefficient to calculate connectivity among nodes (embedded vectors)
    """

    def __init__(self,
                 name,
                 selection_function,
                 guess_range,
                 solution=-1
                 ):
        """
        :param selection_function: takes a value and a guess_guess as input, returns 0 or 1
        :param guess_range: what are the values for the guess guess
        :param solution: if known, indicate the correct guess guess.
        """
        self.BaseGraph = BaseGraph()
        self.solution = solution
        GuessEngine.__init__(self, name, selection_function, guess_range, solution)
        self.logger.debug(
            'Creating A_GraphConstructionTraceAllCorr "%s" with %d guesses.', name, len(guess_range)
        )

    def _initialize(self):
        self.number_of_nodes = self._session.leakage_shape[0]
        self._bits = [1, 2, 4, 8, 16, 32, 64, 128]
        self.l = np.zeros((len(self._bits), self._number_of_guesses, 2, self.number_of_nodes), np.double)
        self.l2 = np.zeros((len(self._bits), self._number_of_guesses, 2, self.number_of_nodes), np.double)
        self.ll = np.zeros((len(self._bits), self._number_of_guesses, 2, self.number_of_nodes, self.number_of_nodes), np.double)
        self._count = np.zeros((len(self._bits), self._number_of_guesses, 2,), np.double)
        self.results = np.zeros((len(self._bits), self._number_of_guesses))
        if self.jit:
            try:
                from numba import jit, uint32
            except Exception:
                raise Exception(
                    "Cannot jit without Numba. Please install Numba or consider turning off the jit option"
                )

            try:
                f = jit(nopython=True)(self._function)
            except Exception:
                raise Exception(
                    "Numba could not jit this guess function. If it contains an assignment such as `value['your_string']`, Numba most likely cannot resolve it. Use the 'value_section' field of your container instead and set it to 'your_string'."
                )

            @jit(nopython=True)
            def hf(batchvalues, guess_range, bit_range):
                out = np.zeros(
                    (batchvalues.shape[0], guess_range.shape[0], bit_range.shape[0]), dtype=np.uint32
                )
                for b in np.arange(bit_range.shape[0]):
                    for guess in np.arange(guess_range.shape[0]):
                        for v in np.arange(batchvalues.shape[0]):
                            out[b, guess, v] = f(batchvalues[v], guess_range[guess], bit_range[b])
                return out
            self._map_function = hf

    def _update(self, batch):
        y = self._map_function(batch.values, self._guess_range, np.array(self._bits))
        lk = batch.leakages
        for guess_i in range(len(self._guess_range)):
            for bit_i in range(len(self._bits)):
                yi = y[bit_i, guess_i, :].flatten()
                for idx, i in enumerate([0, self._bits[bit_i]]):
                    lki = lk[yi == i]
                    self.l[bit_i, guess_i, idx] += lki.sum(0)
                    self.l2[bit_i, guess_i, idx] += (lki ** 2).sum(0)
                    self.ll[bit_i, guess_i, idx] += np.dot(lki.T, lki)
                    self._count[bit_i, guess_i, idx] += lki.shape[0]


class A_GraphConstructionTraceAllDist(GuessEngine):
    """
    convert all traces in to one graph, each column (along trace axis) represents one node
    using Kolmogorov–Smirnov (K-S) statistics to calculate connectivity among nodes (embedded vectors)
    """

    def __init__(self,
                 name,
                 selection_function,
                 guess_range,
                 num_bins=100,
                 solution=-1
                 ):
        """
        :param selection_function: takes a value and a guess_guess as input, returns 0 or 1
        :param guess_range: what are the values for the guess guess
        :param solution: if known, indicate the correct guess guess.
        """
        self.num_bins = num_bins
        self.solution = solution

        self.BaseGraph = BaseGraph()

        GuessEngine.__init__(self, name, selection_function, guess_range, solution)
        self.logger.debug(
            'Creating A_GraphConstructionTraceAllDist "%s" with %d guesses.', name, len(guess_range)
        )

    def _calc_histogram(self, bit_idx, guess_idx, idx, lk, min_data, max_data):
        """
        this update_hist function directly update the previous histogram based on the current data
        it may involve padding operations
        """
        sample_size = lk.shape[0]
        self._count[bit_idx, guess_idx, idx] += sample_size

        lk_flat = lk.flatten(order='F')
        self._update_ranges_counts(bit_idx, guess_idx, idx, min_data, max_data)
        counts_index = np.searchsorted(self.hist_ranges[bit_idx][guess_idx], lk_flat, side='left')
        counts_index[lk_flat != self.hist_ranges[bit_idx][guess_idx][0]] -= 1

        time_sample_idx = 0
        num_bins = len(self.hist_ranges[bit_idx][guess_idx])
        for i, c_idx in enumerate(counts_index):
            if i > 0 and i % sample_size == 0:
                time_sample_idx += 1
            self.hist_counts[bit_idx][guess_idx][idx][c_idx + time_sample_idx * num_bins] += 1

    def _update_ranges_counts(self, bit_idx, guess_idx, idx, min_data, max_data):
        if not isinstance(self.hist_counts[bit_idx][guess_idx][idx], np.ndarray):
            bin_width = (max_data - min_data) / self.num_bins
            self.hist_ranges[bit_idx][guess_idx] = [min_data + i * bin_width for i in range(self.num_bins)]
            one_hist = [0] * len(self.hist_ranges[bit_idx][guess_idx])
            self.hist_counts[bit_idx][guess_idx][idx] = one_hist * self.number_of_nodes
        else:
            bin_width = np.diff(self.hist_ranges[bit_idx][guess_idx])[0]
            min_boundary, max_boundary = np.min(self.hist_ranges[bit_idx][guess_idx]), np.max(self.hist_ranges[bit_idx][guess_idx])
            left_pad, right_pad = list(), list()
            while min_data < min_boundary:
                min_boundary = min_boundary - bin_width
                left_pad.append(min_boundary)
            while max_data - bin_width > max_boundary:
                max_boundary = max_boundary + bin_width
                right_pad.append(max_boundary)
            self.hist_ranges[bit_idx][guess_idx] = left_pad[::-1] + self.hist_ranges[bit_idx][guess_idx] + right_pad
            old_hist_counts = np.array(self.hist_counts[bit_idx][guess_idx][idx])
            to_mat = np.reshape(old_hist_counts, (-1, self.number_of_nodes), order='F')
            updated_hist_counts = np.pad(to_mat, ((len(left_pad), len(right_pad)), (0, 0)), 'constant', constant_values=(0, 0))
            self.hist_counts[bit_idx][guess_idx][idx] = updated_hist_counts.flatten(order='F')

    def _initialize(self):
        self.number_of_nodes = self._session.leakage_shape[0]
        self._bits = [1, 2, 4, 8, 16, 32, 64, 128]
        self.hist_ranges = [[None] * self._number_of_guesses for _ in range(len(self._bits))]
        self.hist_counts = [[[None] * 2 for _ in range(self._number_of_guesses)] for _ in range(len(self._bits))]
        self._count = np.zeros((len(self._bits), self._number_of_guesses, 2,), np.double)
        self.results = np.zeros((len(self._bits), self._number_of_guesses))
        if self.jit:
            try:
                from numba import jit, uint32
            except Exception:
                raise Exception(
                    "Cannot jit without Numba. Please install Numba or consider turning off the jit option"
                )

            try:
                f = jit(nopython=True)(self._function)
            except Exception:
                raise Exception(
                    "Numba could not jit this guess function. If it contains an assignment such as `value['your_string']`, Numba most likely cannot resolve it. Use the 'value_section' field of your container instead and set it to 'your_string'."
                )

            @jit(nopython=True)
            def hf(batchvalues, guess_range, bit_range):
                x = list()
                x.append(0)
                out = np.zeros(
                    (bit_range.shape[0], guess_range.shape[0], batchvalues.shape[0]), dtype=np.uint32
                )
                for b in np.arange(bit_range.shape[0]):
                    for guess in np.arange(guess_range.shape[0]):
                        for v in np.arange(batchvalues.shape[0]):
                            out[b, guess, v] = f(batchvalues[v], guess_range[guess], bit_range[b])
                return out
            self._map_function = hf

    def _update(self, batch):
        y = self._map_function(batch.values, self._guess_range, np.array(self._bits))
        lk = batch.leakages
        for guess_i in range(len(self._guess_range)):
            for bit_i in range(len(self._bits)):
                yi = y[bit_i, guess_i, :].flatten()
                for idx, i in enumerate([0, self._bits[bit_i]]):
                    lki = lk[yi == i]
                    min_lki, max_lki = np.min(lki), np.max(lki)
                    self._calc_histogram(bit_i, guess_i, idx, lki, min_lki, max_lki)
                    self._count[bit_i, guess_i, 0] += lki.shape[0]
