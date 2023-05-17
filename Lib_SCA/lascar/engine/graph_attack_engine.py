from scipy.stats import binom
from tqdm import tqdm
import numpy as np

from . import A_GraphConstructionPSR, A_GraphConstructionAB, A_GraphConstructionTraceBatch
from . import A_GraphConstructionTraceAllCorr_SB, A_GraphConstructionTraceAllHistogram_SB, \
    A_GraphConstructionTraceAllCorr_MB, A_GraphConstructionTraceAllHistogram_MB


class GraphAttackEngineTraceAllCorr_SB(A_GraphConstructionTraceAllCorr_SB):
    """
    dpa type of attack, using spectral distance (eigenvalue distance) as measurement
    """

    def __init__(self,
                 name,
                 selection_function,
                 guess_range,
                 solution=-1,
                 dist_type='edit_dist',
                 sd_params=None,
                 jit=True
                 ):
        self.dist_type = dist_type
        self.sd_params = sd_params
        A_GraphConstructionTraceAllCorr_SB.__init__(self, name, selection_function, guess_range, dist_type, solution, jit)

    def _finalize(self):
        # return self._finalize_function()
        print('[INFO] calculating results...')
        for bit in tqdm(range(len(self._bits))):
            for guess in range(self._number_of_guesses):
                m_r, m_f = self._count[bit, guess, 0], self._count[bit, guess, 1]
                self.l[bit, guess, 0], self.l[bit, guess, 1] = self.l[bit, guess, 0] / m_r, self.l[bit, guess, 1] / m_f
                self.l2[bit, guess, 0], self.l2[bit, guess, 1] = self.l2[bit, guess, 0] / m_r, self.l2[
                    bit, guess, 1] / m_f
                self.ll[bit, guess, 0], self.ll[bit, guess, 1] = self.ll[bit, guess, 0] / m_r, self.ll[
                    bit, guess, 1] / m_f

                v_r = self.l2[bit, guess, 0] - self.l[bit, guess, 0] ** 2
                v_f = self.l2[bit, guess, 1] - self.l[bit, guess, 1] ** 2
                numerator_r = self.ll[bit, guess, 0] - np.outer(self.l[bit, guess, 0], self.l[bit, guess, 0])
                numerator_f = self.ll[bit, guess, 1] - np.outer(self.l[bit, guess, 1], self.l[bit, guess, 1])
                denominator_r, denominator_f = np.sqrt(np.outer(v_r, v_r)), np.sqrt(np.outer(v_f, v_f))
                mask_r, mask_f = v_r == 0.0, v_f == 0.0
                numerator_r[mask_r, mask_r], denominator_r[mask_r, mask_r] = 0.0, 1.0
                numerator_f[mask_f, mask_f], denominator_f[mask_f, mask_f] = 0.0, 1.0
                graph_samples_r = np.abs(np.nan_to_num(numerator_r / denominator_r))
                graph_samples_f = np.abs(np.nan_to_num(numerator_f / denominator_f))
                if self.dist_type == 'spectral_dist':
                    self.results[bit, guess] = self.graph_distance(graph_samples_r, graph_samples_f, params=self.sd_params)
                else:
                    self.results[bit, guess] = self.graph_distance(graph_samples_r, graph_samples_f)
        self.results = self.results.T
        return self.results

    def _clean(self):
        del self.l
        del self.l2
        del self.ll
        del self._count
        from numba import cuda
        cuda.current_context().reset()


class GraphAttackEngineTraceAllCorr_MB(A_GraphConstructionTraceAllCorr_MB):
    """
    dpa type of attack, using spectral distance (eigenvalue distance) as measurement
    """

    def __init__(self,
                 name,
                 selection_function,
                 guess_range,
                 solution=-1,
                 dist_type='edit_dist',
                 sd_params=None,
                 jit=True
                 ):
        self.dist_type = dist_type
        self.sd_params = sd_params
        A_GraphConstructionTraceAllCorr_MB.__init__(self, name, selection_function, guess_range, dist_type, solution, jit)

    def _finalize(self):
        from numba import cuda
        cuda.current_context().reset()
        ## calculate real leakage graph
        l_all = self.l_all / self._count_all
        l2_all = self.l2_all / self._count_all
        ll_all = self.ll_all / self._count_all
        v_all = l2_all - l_all ** 2
        numerator_all = ll_all - np.outer(l_all, l_all)
        denominator_all = np.sqrt(np.outer(v_all, v_all))
        mask_all = denominator_all == 0.0
        numerator_all[mask_all], denominator_all[mask_all] = 0.0, 1.0
        graph_all = np.abs(np.nan_to_num(numerator_all / denominator_all))

        print('[INFO] calculating results...')
        weights = binom.pmf([0, 1, 2, 3, 4, 5, 6, 7, 8], 8, 0.5)
        for guess in range(self._number_of_guesses):
            graph_dists = np.zeros(9)
            for i in range(9):
                m = self._count[guess, i]
                li = self.l[guess, i] / m
                l2i = self.l2[guess, i] / m
                lli = self.ll[guess, i] / m
                v = l2i - li ** 2
                numerator = lli - np.outer(li, li)
                denominator = np.sqrt(np.outer(v, v))
                mask = denominator == 0.0
                numerator[mask], denominator[mask] = 0.0, 1.0
                graph_sample = np.abs(np.nan_to_num(numerator / denominator))
                if self.dist_type == 'spectral_dist':
                    graph_dists[i] = self.graph_distance(graph_sample, graph_all, params=self.sd_params)
                else:
                    graph_dists[i] = self.graph_distance(graph_sample, graph_all)
            self.results[guess] = np.sum(graph_dists * weights)
        return self.results

    def _clean(self):
        del self.l
        del self.l2
        del self.ll
        del self._count


class GraphAttackEngineTraceAllKSDist_SB(A_GraphConstructionTraceAllHistogram_SB):
    """
    dpa type attack for single bit
    convert all traces in to one graph, each column (along trace axis) represents one node
    using Kolmogorov–Smirnov (K-S) statistics to calculate connectivity among nodes (embedded vectors)
    """

    def __init__(self,
                 name,
                 selection_function,
                 guess_range,
                 num_bins=100,
                 dist_type='edit_dist',
                 sd_params=None,
                 solution=-1,
                 jit=True
                 ):
        self.dist_type = dist_type
        self.sd_params = sd_params
        A_GraphConstructionTraceAllHistogram_SB.__init__(self, name, selection_function, guess_range, num_bins, dist_type, solution, jit)

    def _finalize(self):
        # return self._finalize_function()
        print('[INFO] calculating results...')
        for bit in tqdm(range(len(self._bits))):
            for guess in range(self._number_of_guesses):
                m_r, m_f = self._count[bit, guess, 0], self._count[bit, guess, 1]
                pdf_r, pdf_f = np.array(self.hist_counts[bit][guess][0]) / m_r, np.array(
                    self.hist_counts[bit][guess][1]) / m_f
                pdf_r_with_matrix = np.reshape(pdf_r, (-1, self.number_of_nodes), order='F')
                pdf_f_with_matrix = np.reshape(pdf_f, (-1, self.number_of_nodes), order='F')
                cdf_r_with_matrix = np.cumsum(pdf_r_with_matrix, axis=0)
                cdf_f_with_matrix = np.cumsum(pdf_f_with_matrix, axis=0)

                # more large the weight is, more close two nodes are
                graph_samples_r = self._calc_kolmogorov_smirnov_distance(cdf_r_with_matrix, self.number_of_nodes)
                graph_samples_f = self._calc_kolmogorov_smirnov_distance(cdf_f_with_matrix, self.number_of_nodes)
                if self.dist_type == 'spectral_dist':
                    self.results[bit, guess] = self.graph_distance(graph_samples_r, graph_samples_f, params=self.sd_params)
                else:
                    self.results[bit, guess] = self.graph_distance(graph_samples_r, graph_samples_f)
        self.results = self.results.T
        return self.results

    def _clean(self):
        del self.hist_ranges
        del self.hist_counts
        del self._count
        from numba import cuda
        cuda.current_context().reset()

    @staticmethod
    def _calc_kolmogorov_smirnov_distance(cdf_with_matrix, num_nodes):
        # try to construct to #nodes * #nodes size of column vectors to speedup the calculation
        tmp1 = np.repeat(cdf_with_matrix, num_nodes, axis=1)
        tmp2 = np.tile(cdf_with_matrix, num_nodes)
        cdf_diff = np.abs(tmp1 - tmp2)
        ks_dist = np.max(cdf_diff, axis=0)  # size is #nodes * #nodes
        adj_matrix = np.reshape(ks_dist, (num_nodes, num_nodes), order='C')

        # # numba version
        # cwm_f = cdf_with_matrix.flatten()
        # cwn_fr = np.repeat(cwm_f, num_nodes)
        # tmp1 = np.reshape(cwn_fr, (-1, num_nodes * num_nodes))
        # tmp2 = cdf_with_matrix
        # for i in range(num_nodes-1):
        #     tmp2 = np.concatenate((tmp2, cdf_with_matrix), axis=1)
        #
        # # max in numba has no axis
        # cdf_diff = np.abs(tmp1 - tmp2)
        # ks_dist = np.zeros(num_nodes * num_nodes)
        # for i in range(num_nodes * num_nodes):
        #     ks_dist[i] = np.max(cdf_diff[:, i])  # size is #nodes * #nodes
        # adj_matrix = np.reshape(ks_dist, (num_nodes, num_nodes))
        return adj_matrix


class GraphAttackEngineTraceAllChi2_SB(A_GraphConstructionTraceAllHistogram_SB):
    """
    dpa type attack for single bit (8)
    convert all traces in to one graph, each column (along trace axis) represents one node
    using chi2 statistics to calculate connectivity among nodes (embedded vectors)
    """

    def __init__(self,
                 name,
                 selection_function,
                 guess_range,
                 num_bins=100,
                 dist_type='edit_dist',
                 sd_params=None,
                 solution=-1,
                 jit=True
                 ):
        self.dist_type = dist_type
        self.sd_params = sd_params
        A_GraphConstructionTraceAllHistogram_SB.__init__(self, name, selection_function, guess_range, num_bins, dist_type, solution, jit)

    def _finalize(self):
        from numba import cuda
        cuda.current_context().reset()
        print('[INFO] calculating results...')
        for bit in tqdm(range(len(self._bits))):
            for guess in range(self._number_of_guesses):
                hist_r, hist_f = self.hist_counts[bit][guess][0], self.hist_counts[bit][guess][1]
                hist_r_with_matrix = np.reshape(hist_r, (-1, self.number_of_nodes), order='F')
                hist_f_with_matrix = np.reshape(hist_f, (-1, self.number_of_nodes), order='F')
                graph_samples_r = self._chi2graph(hist_r_with_matrix)
                graph_samples_f = self._chi2graph(hist_f_with_matrix)
                if self.dist_type == 'spectral_dist':
                    self.results[bit, guess] = self.graph_distance(graph_samples_r, graph_samples_f, params=self.sd_params)
                else:
                    self.results[bit, guess] = self.graph_distance(graph_samples_r, graph_samples_f)
        self.results = self.results.T
        return self.results

    def _clean(self):
        del self.hist_ranges
        del self.hist_counts
        del self._count

    def _chi2graph(self, hist):
        num_bins = self.hist_ranges.shape[0]
        hist = hist.T
        tmp1 = np.repeat(hist, self.number_of_nodes, axis=0)
        tmp2 = np.tile(hist, (self.number_of_nodes, 1))
        dummy = np.repeat(tmp1, 2, axis=0)
        dummy[1::2, :] = tmp2
        chi2table_3d = np.reshape(dummy, (self.number_of_nodes ** 2, 2, num_bins), order='C')

        col_sum = np.sum(chi2table_3d, axis=1, keepdims=True)
        row_sum = np.sum(chi2table_3d, axis=2, keepdims=True)
        n = np.sum(chi2table_3d, axis=(1, 2), keepdims=True)
        expected_freq = (row_sum @ col_sum) / n

        numerator = (chi2table_3d - expected_freq) ** 2
        chi2_terms = np.divide(numerator, expected_freq, out=np.zeros_like(numerator), where=expected_freq != 0)
        chi2_score = np.sum(chi2_terms, axis=(1, 2))
        adj = np.reshape(chi2_score, (self.number_of_nodes, self.number_of_nodes), order='C')

        return adj


class GraphAttackEngineTraceAllChi2_MB(A_GraphConstructionTraceAllHistogram_MB):
    """
    dpa type attack for multiple bits
    convert all traces in to one graph, each column (along trace axis) represents one node
    using chi2 statistics to calculate connectivity among nodes (embedded vectors)
    """

    def __init__(self,
                 name,
                 selection_function,
                 guess_range,
                 num_bins=100,
                 dist_type='edit_dist',
                 sd_params=None,
                 solution=-1,
                 jit=True
                 ):
        self.dist_type = dist_type
        self.sd_params = sd_params
        A_GraphConstructionTraceAllHistogram_MB.__init__(self, name, selection_function, guess_range, num_bins, dist_type,
                                                         solution, jit)

    def _finalize(self):
        from numba import cuda
        cuda.current_context().reset()
        hist_all = self.hist_counts_all
        hist_all_with_matrix = np.reshape(hist_all, (-1, self.number_of_nodes), order='F')
        graph_all = self._chi2graph(hist_all_with_matrix)

        print('[INFO] calculating results...')
        weights = binom.pmf([0, 1, 2, 3, 4, 5, 6, 7, 8], 8, 0.5)
        for guess in range(self._number_of_guesses):
            graph_dists = np.zeros(9)
            for i in range(9):
                hist = self.hist_counts[guess][i]
                hist_with_matrix = np.reshape(hist, (-1, self.number_of_nodes), order='F')
                graph_sample = self._chi2graph(hist_with_matrix)
                if self.dist_type == 'spectral_dist':
                    graph_dists[i] = self.graph_distance(graph_sample, graph_all, params=self.sd_params)
                else:
                    graph_dists[i] = self.graph_distance(graph_sample, graph_all)
            self.results[guess] = np.sum(graph_dists * weights)
        return self.results

    def _clean(self):
        del self.hist_ranges
        del self.hist_counts
        del self._count

    def _chi2graph(self, hist):
        num_bins = self.hist_ranges.shape[0]
        hist = hist.T
        tmp1 = np.repeat(hist, self.number_of_nodes, axis=0)
        tmp2 = np.tile(hist, (self.number_of_nodes, 1))
        dummy = np.repeat(tmp1, 2, axis=0)
        dummy[1::2, :] = tmp2
        chi2table_3d = np.reshape(dummy, (self.number_of_nodes ** 2, 2, num_bins), order='C')

        col_sum = np.sum(chi2table_3d, axis=1, keepdims=True)
        row_sum = np.sum(chi2table_3d, axis=2, keepdims=True)
        n = np.sum(chi2table_3d, axis=(1, 2), keepdims=True)
        expected_freq = (row_sum @ col_sum) / n

        numerator = (chi2table_3d - expected_freq) ** 2
        chi2_terms = np.divide(numerator, expected_freq, out=np.zeros_like(numerator), where=expected_freq != 0)
        chi2_score = np.sum(chi2_terms, axis=(1, 2))
        adj = np.reshape(chi2_score, (self.number_of_nodes, self.number_of_nodes), order='C')

        return adj

# class GraphAttackEngineTraceBatch(A_GraphConstructionTraceBatch):
#     def __init__(self,
#                  name,
#                  selection_function,
#                  guess_range,
#                  solution=-1,
#                  graph_type='corr',
#                  k=100,
#                  mode='l2',
#                  num_bins=100
#                  ):
#         """
#         :param selection_function: takes a value and a guess_guess as input, returns 0 or 1
#         :param guess_range: what are the values for the guess guess
#         :param solution: if known, indicate the correct guess guess.
#         :param k: k largest eigenvalues to be compared
#         :param mode: 'l2', 'lmax', 'ks' --> Kolmogorov–Smirnov (K-S) distance/statistic
#         """
#         self.graph_type = graph_type
#         self.num_bins = num_bins
#         self.sd = SpectralDistance(mode, k)
#         A_GraphConstructionTraceBatch.__init__(self, name, selection_function, guess_range, solution)
#
#     # def _finalize(self):
#     #     self.number_of_nodes = self._number_of_processed_traces // self.batch_size
#     #     graph_samples_guess = []
#     #     for gi in tqdm(range(self._number_of_guesses)):
#     #         if self.graph_type == 'corr':
#     #             graph_samples_guess.append(np.abs(np.corrcoef(self.embeddings[gi + 1])))
#     #         elif self.graph_type == 'dist':
#     #             graph_samples_guess.append(self._calc_ks_graph(self.embeddings[gi + 1]))
#     #
#     #     for ti in tqdm(range(self.number_of_time_samples)):
#     #         lk_real = self.embeddings[0][:, ti * self.batch_size: (ti + 1) * self.batch_size]
#     #         if self.graph_type == 'corr':
#     #             graph_samples_real = np.abs(np.corrcoef(lk_real))
#     #         elif self.graph_type == 'dist':
#     #             graph_samples_real = self._calc_ks_graph(lk_real)
#     #         for gi in range(self._number_of_guesses):
#     #             # self.results[gi, ti] = nc.lambda_dist(graph_samples_real, graph_samples_guess[gi], k=100, kind='laplacian_norm')
#     #             self.results[gi, ti] = nc.edit_distance(graph_samples_real, graph_samples_guess[gi])
#     #             # self.results[gi, ti] = self.sd.spectral_distance(graph_samples_real, graph_samples_guess[gi])
#     #     return self.results
#
#     def _finalize(self):
#         self.number_of_nodes = self._number_of_processed_traces // self.batch_size
#         for ti in tqdm(range(self.number_of_time_samples)):
#             lk_real = self.embeddings[0][:, ti * self.batch_size: (ti + 1) * self.batch_size]
#             graph_samples_real = np.corrcoef(lk_real, lk_real)
#             for gi in range(self._number_of_guesses):
#                 graph_samples_guess = np.corrcoef(self.embeddings[gi + 1], lk_real)
#                 # graph_samples_guess = tmp[:self.number_of_nodes, self.number_of_nodes:]
#                 # self.results[gi, ti] = self.sd.spectral_distance(graph_samples_real, graph_samples_guess)
#                 self.results[gi, ti] = nc.resistance_distance(graph_samples_real, graph_samples_guess)
#         return self.results
#
#     def _calc_ks_graph(self, data):
#         min_data, max_data = np.min(data), np.max(data)
#         bin_width = (max_data - min_data) / self.num_bins
#         hist_ranges = [min_data + i * bin_width for i in range(self.num_bins)]
#         one_hist = [0] * len(hist_ranges)
#         hist_counts = one_hist * self.number_of_nodes
#         data_flat = data.flatten()
#         counts_index = np.searchsorted(hist_ranges, data_flat, side='left')
#         counts_index[data_flat != hist_ranges[0]] -= 1
#         batch_idx = 0
#         num_bins = len(hist_ranges)
#         for i, c_idx in enumerate(counts_index):
#             if i > 0 and i % self.batch_size == 0:
#                 batch_idx += 1
#             hist_counts[c_idx + batch_idx * num_bins] += 1
#         pdf = np.array(hist_counts) / self.batch_size
#         pdf_with_matrix = np.reshape(pdf, (self.number_of_nodes, num_bins)).T
#         cdf_with_matrix = np.cumsum(pdf_with_matrix, axis=0)
#         # more large the weight is, more close two nodes are
#         graph_samples = self._calc_kolmogorov_smirnov_distance(cdf_with_matrix)
#         return graph_samples
#
#     def _calc_kolmogorov_smirnov_distance(self, cdf_with_matrix):
#         # try to construct to #nodes * #nodes size of column vectors to speedup the calculation
#         tmp1 = np.repeat(cdf_with_matrix, self.number_of_nodes, axis=1)
#         tmp2 = np.tile(cdf_with_matrix, self.number_of_nodes)
#         cdf_diff = np.abs(tmp1 - tmp2)
#         ks_dist = np.max(cdf_diff, axis=0)  # size is #nodes * #nodes
#         adj_matrix = np.reshape(ks_dist, (self.number_of_nodes, self.number_of_nodes), order='C')
#         return adj_matrix
