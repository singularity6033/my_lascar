"""
GraphTestEngine is a two-samples graph testing used to compute whether two graph samples G and H are drawn from
the same distribution or not
"""
import collections
from collections import defaultdict
import numpy as np
from .graph_attack_basic import GraphDistance

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
                self.embeddings[guess_i + 1] = np.concatenate((self.embeddings[guess_i + 1], yi), axis=0)


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
                 mode,
                 k,
                 solution=-1,
                 jit=True
                 ):
        """
        :param selection_function: takes a value and a guess_guess as input, returns 0 or 1
        :param guess_range: what are the values for the guess guess
        :param mode & k: used in graph distance
        :param solution: if known, indicate the correct guess guess
        :param jit: if true, using numba to acceleration, if your computer doesn't have gpu pls set it to false
        """
        self.BaseGraph = BaseGraph()
        self.mode = mode
        self.k = k
        self.solution = solution
        self.graph_distance = GraphDistance(mode, k).graph_distance
        GuessEngine.__init__(self, name, selection_function, guess_range, solution, jit)
        self.logger.debug(
            'Creating A_GraphConstructionTraceAllCorr "%s" with %d guesses.', name, len(guess_range)
        )

    def _initialize(self):
        self.number_of_nodes = self._session.leakage_shape[0]
        self._bits = [1, 2, 4, 8, 16, 32, 64, 128]
        self.l = np.zeros((len(self._bits), self._number_of_guesses, 2, self.number_of_nodes), np.double)
        self.l2 = np.zeros((len(self._bits), self._number_of_guesses, 2, self.number_of_nodes), np.double)
        self.ll = np.zeros((len(self._bits), self._number_of_guesses, 2, self.number_of_nodes, self.number_of_nodes),
                           np.double)
        self._count = np.zeros((len(self._bits), self._number_of_guesses, 2,), np.double)
        self.results = np.zeros((len(self._bits), self._number_of_guesses))
        if self.jit:
            try:
                from numba import jit, uint32, cuda
            except Exception:
                raise Exception(
                    "Cannot jit without Numba. Please install Numba or consider turning off the jit option"
                )

            try:
                cuda.select_device(0)
                f = jit(nopython=True)(self._function)
                # gd = jit(nopython=True)(self.graph_distance)
            except Exception:
                raise Exception(
                    "Numba could not jit this guess function. If it contains an assignment such as `value['your_string']`, Numba most likely cannot resolve it. Use the 'value_section' field of your container instead and set it to 'your_string'."
                )

            @jit(nopython=True)
            def _dpa_partition(batchvalues, guess_range=self._guess_range, bit_range=np.array(self._bits)):
                out = np.zeros(
                    (bit_range.shape[0], guess_range.shape[0], batchvalues.shape[0]), dtype=np.uint32
                )
                for b in np.arange(bit_range.shape[0]):
                    for guess in np.arange(guess_range.shape[0]):
                        for v in np.arange(batchvalues.shape[0]):
                            out[b, guess, v] = f(batchvalues[v], guess_range[guess], bit_range[b])
                return out

            @jit(nopython=True)
            def _increment_term(pv, batchleakages, guess_range=self._guess_range, bit_range=np.array(self._bits),
                                num_node=self.number_of_nodes):
                l = np.zeros((bit_range.shape[0], guess_range.shape[0], 2, num_node), np.double)
                l2 = np.zeros((bit_range.shape[0], guess_range.shape[0], 2, num_node), np.double)
                ll = np.zeros((bit_range.shape[0], guess_range.shape[0], 2, num_node, num_node), np.double)
                count = np.zeros((bit_range.shape[0], guess_range.shape[0], 2,), np.double)
                lk = batchleakages
                for guess_i in np.arange(guess_range.shape[0]):
                    for bit_i in np.arange(bit_range.shape[0]):
                        pvi = pv[bit_i, guess_i, :].flatten()
                        for idx, i in enumerate([0, bit_range[bit_i]]):
                            lki = lk[pvi == i]
                            l[bit_i, guess_i, idx] += lki.sum(0)
                            l2[bit_i, guess_i, idx] += (lki ** 2).sum(0)
                            ll[bit_i, guess_i, idx] += np.dot(lki.T, lki)
                            count[bit_i, guess_i, idx] += lki.shape[0]
                return l, l2, ll, count

            # @jit(nopython=True)
            # def _calc_corr(l=self.l, l2=self.l2, ll=self.ll, count=self._count, guess_range=self._guess_range,
            #                bit_range=np.array(self._bits), k=self.k):
            #     results = np.zeros((bit_range.shape[0], guess_range.shape[0]))
            #     print('[INFO] calculating results...')
            #     for bit in range(bit_range.shape[0]):
            #         for guess in range(guess_range.shape[0]):
            #             m_r, m_f = count[bit, guess, 0], count[bit, guess, 1]
            #             l[bit, guess, 0], l[bit, guess, 1] = l[bit, guess, 0] / m_r, l[bit, guess, 1] / m_f
            #             l2[bit, guess, 0], l2[bit, guess, 1] = l2[bit, guess, 0] / m_r, l2[bit, guess, 1] / m_f
            #             ll[bit, guess, 0], ll[bit, guess, 1] = ll[bit, guess, 0] / m_r, ll[bit, guess, 1] / m_f
            #
            #             v_r = l2[bit, guess, 0] - l[bit, guess, 0] ** 2
            #             v_f = l2[bit, guess, 1] - l[bit, guess, 1] ** 2
            #             numerator_r = ll[bit, guess, 0] - np.outer(l[bit, guess, 0], l[bit, guess, 0])
            #             numerator_f = ll[bit, guess, 1] - np.outer(l[bit, guess, 1], l[bit, guess, 1])
            #             denominator_r, denominator_f = np.sqrt(np.outer(v_r, v_r)), np.sqrt(np.outer(v_f, v_f))
            #
            #             mask_r, mask_f = np.where(denominator_r == 0.0), np.where(denominator_f == 0.0)
            #             for i, j in zip(mask_r[0], mask_r[1]):
            #                 numerator_r[i, j], denominator_r[i, j] = 0.0, 1.0
            #             for i, j in zip(mask_f[0], mask_f[1]):
            #                 numerator_f[i, j], denominator_f[i, j] = 0.0, 1.0
            #
            #             # mask_r, mask_f = np.where(denominator_r == 0.0), np.where(denominator_f == 0.0)
            #             # numerator_r[mask_r], denominator_r[mask_r] = 0.0, 1.0
            #             # numerator_f[mask_f], denominator_f[mask_f] = 0.0, 1.0
            #
            #             graph_samples_r = np.abs(numerator_r / denominator_r)
            #             graph_samples_f = np.abs(numerator_f / denominator_f)
            #             # graph_samples_r = np.abs(np.nan_to_num(numerator_r / denominator_r))
            #             # graph_samples_f = np.abs(np.nan_to_num(numerator_f / denominator_f))
            #             results[bit, guess] = gd(graph_samples_r, graph_samples_f, k)
            #     results = results.T
            #     return results

            self._map_function = _dpa_partition
            self._update_function = _increment_term
            # self._finalize_function = _calc_corr

    def _update(self, batch):
        y = self._map_function(batch.values)

        # numba gpu acceleration
        if self.jit:
            l, l2, ll, count = self._update_function(y, batch.leakages)
            self.l += l
            self.l2 += l2
            self.ll += ll
            self._count += count

        # traditional version
        else:
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
                 num_bins,
                 mode,
                 k,
                 solution=-1,
                 jit=True
                 ):
        """
        :param selection_function: takes a value and a guess_guess as input, returns 0 or 1
        :param guess_range: what are the values for the guess guess
        :param num_bins: used in ks calculation
        :param mode & k: used in graph distance
        :param solution: if known, indicate the correct guess guess
        :param jit: if true, using numba to acceleration, if your computer doesn't have gpu pls set it to false
        """
        self.num_bins = num_bins
        self.mode = mode
        self.k = k
        self.solution = solution
        self.graph_distance = GraphDistance(mode, k).graph_distance
        self.BaseGraph = BaseGraph()

        GuessEngine.__init__(self, name, selection_function, guess_range, solution, jit)
        self.logger.debug(
            'Creating A_GraphConstructionTraceAllDist "%s" with %d guesses.', name, len(guess_range)
        )

    def _initialize(self):
        self.number_of_nodes = self._session.leakage_shape[0]
        self._bits = [1, 2, 4, 8, 16, 32, 64, 128]
        self.hist_ranges = np.empty(self.num_bins)
        self.hist_counts = np.zeros(
            (len(self._bits), self._number_of_guesses, 2, self.hist_ranges.shape[0] * self.number_of_nodes))
        self._count = np.zeros((len(self._bits), self._number_of_guesses, 2,), np.double)
        self.results = np.zeros((len(self._bits), self._number_of_guesses))
        if self.jit:
            try:
                from numba import jit, uint32, cuda
            except Exception:
                raise Exception(
                    "Cannot jit without Numba. Please install Numba or consider turning off the jit option"
                )

            try:
                cuda.select_device(3)
                f = jit(nopython=True)(self._function)
                ch = jit(nopython=True)(self._calc_histogram)
                # gd = jit(nopython=True)(self.graph_distance)
                # ks = jit(nopython=True)(self._calc_kolmogorov_smirnov_distance)
            except Exception:
                raise Exception(
                    "Numba could not jit this guess function. If it contains an assignment such as `value['your_string']`, Numba most likely cannot resolve it. Use the 'value_section' field of your container instead and set it to 'your_string'."
                )

            @jit(nopython=True)
            def _dpa_partition(batchvalues, guess_range, bit_range):
                out = np.zeros(
                    (bit_range.shape[0], guess_range.shape[0], batchvalues.shape[0]), dtype=np.uint32
                )
                for b in np.arange(bit_range.shape[0]):
                    for guess in np.arange(guess_range.shape[0]):
                        for v in np.arange(batchvalues.shape[0]):
                            out[b, guess, v] = f(batchvalues[v], guess_range[guess], bit_range[b])
                return out

            @jit(nopython=True)
            def _increment_hist(pv, batchleakages, hist_ranges, hist_counts, guess_range=self._guess_range,
                                bit_range=np.array(self._bits), num_bins=self.num_bins, num_nodes=self.number_of_nodes):
                count = np.zeros((bit_range.shape[0], guess_range.shape[0], 2,), np.double)
                lk = batchleakages
                min_data, max_data = np.min(lk), np.max(lk)

                # update hist range
                if not hist_ranges.all():
                    bin_width = (max_data - min_data) / num_bins
                    hist_ranges = np.array([min_data + i * bin_width for i in range(num_bins)])
                else:
                    bin_width = np.diff(hist_ranges)[0]
                    min_boundary, max_boundary = np.min(hist_ranges), np.max(hist_ranges)
                    left_pad, right_pad = list(), list()
                    while min_data < min_boundary:
                        min_boundary = min_boundary - bin_width
                        left_pad.append(min_boundary)
                    while max_data - bin_width > max_boundary:
                        max_boundary = max_boundary + bin_width
                        right_pad.append(max_boundary)
                    hist_ranges = np.concatenate((np.array(left_pad[::-1]), hist_ranges, np.array(right_pad)))
                    left_padding = np.zeros((len(bit_range), len(guess_range), 2, num_nodes, len(left_pad)))
                    right_padding = np.zeros((len(bit_range), len(guess_range), 2, num_nodes, len(right_pad)))
                    old_hist_count = np.reshape(hist_counts, (len(bit_range), len(guess_range), 2, num_nodes, -1))
                    updated_hist_count = np.concatenate((left_padding, old_hist_count, right_padding), axis=-1)
                    # # np.pad cannot directly used in numba
                    # updated_hist_count = np.pad(old_hist_count,
                    #                             ((0, 0), (0, 0), (0, 0), (0, 0), (len(left_pad), len(right_pad))),
                    #                             'constant',
                    #                             constant_values=(0, 0))
                    hist_counts = np.reshape(updated_hist_count,
                                             (len(bit_range), len(guess_range), 2, hist_ranges.shape[0] * num_nodes))

                for guess_i in range(guess_range.shape[0]):
                    for bit_i in range(bit_range.shape[0]):
                        pvi = pv[bit_i, guess_i, :].flatten()
                        for idx, i in enumerate([0, bit_range[bit_i]]):
                            lki = lk[pvi == i]
                            hc = ch(hist_ranges, hist_counts[bit_i, guess_i, idx], lki, num_bins)
                            hist_counts[bit_i, guess_i, idx] = hc
                            count[bit_i, guess_i, idx] += lki.shape[0]
                return hist_ranges, hist_counts, count

            # @jit(nopython=True)
            # def _calc_ks_dist(hist_counts=self.hist_counts, count=self._count, guess_range=self._guess_range,
            #                   bit_range=np.array(self._bits), k=self.k, num_nodes=self.number_of_nodes):
            #     results = np.zeros((bit_range.shape[0], guess_range.shape[0]))
            #     print('[INFO] calculating results...')
            #     for bit in range(bit_range.shape[0]):
            #         for guess in range(guess_range.shape[0]):
            #             m_r, m_f = count[bit, guess, 0], count[bit, guess, 1]
            #             pdf_r, pdf_f = hist_counts[bit, guess, 0] / m_r, hist_counts[bit, guess, 1] / m_f
            #             pdf_r_with_matrix = np.reshape(pdf_r, (num_nodes, -1)).T
            #             pdf_f_with_matrix = np.reshape(pdf_f, (num_nodes, -1)).T
            #             cdf_r_with_matrix = np.zeros(pdf_r_with_matrix.shape)
            #             cdf_f_with_matrix = np.zeros(pdf_f_with_matrix.shape)
            #             # cumsum in numba only has np axis
            #             for n_node in range(num_nodes):
            #                 cdf_r_with_matrix[:, n_node] = np.cumsum(pdf_r_with_matrix[:, n_node])
            #                 cdf_f_with_matrix[:, n_node] = np.cumsum(pdf_f_with_matrix[:, n_node])
            #
            #             graph_samples_r = ks(cdf_r_with_matrix, num_nodes)
            #             graph_samples_f = ks(cdf_f_with_matrix, num_nodes)
            #             results[bit, guess] = gd(graph_samples_r, graph_samples_f, k)
            #     return results

            self._map_function = _dpa_partition
            self._update_function = _increment_hist
            # self._finalize_function = _calc_ks_dist

    def _update(self, batch):
        y = self._map_function(batch.values, self._guess_range, np.array(self._bits))
        hr, hc, count = self._update_function(y, batch.leakages, self.hist_ranges, self.hist_counts)
        self.hist_ranges, self.hist_counts = hr, hc
        self._count += count

    @staticmethod
    def _calc_histogram(hist_ranges, hist_count, data, num_bins):
        """
        this update_hist function directly update the previous histogram based on the current data
        it may involve padding operations
        """
        sample_size = data.shape[0]
        data = data.T
        lk_flat = data.flatten()

        count_index = np.searchsorted(hist_ranges, lk_flat, side='left')
        count_index[lk_flat != hist_ranges[0]] -= 1

        time_sample_idx = 0
        for i, c_idx in enumerate(count_index):
            if i > 0 and i % sample_size == 0:
                time_sample_idx += 1
            hist_count[c_idx + time_sample_idx * num_bins] += 1
        return hist_count
