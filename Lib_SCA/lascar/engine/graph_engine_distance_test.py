import gc
from bisect import bisect_left

from tqdm import tqdm
import numpy as np
from scipy.stats import ttest_ind, chisquare, chi2_contingency
from scipy.stats import norm, bernoulli, chi2, ks_2samp, cramervonmises_2samp
from statsmodels.stats.weightstats import ztest
import networkx as nx
import netcomp as nc

from . import PartitionerEngine, GuessEngine, Phase_Space_Reconstruction_Graph, Generalised_RDPG


class GraphDistanceEngine(PartitionerEngine):
    """
        GraphDistanceEngine is used to calculate graph differences by some distance measurements and corresponding statistical testing procedure is also included in this engine
        based on the paper: Wills, Peter, and François G. Meyer. "Metrics for graph comparison: a practitioner’s guide." Plos one 15.2 (2020): e0228728.
    """

    def __init__(self,
                 name,
                 partition_function,
                 time_delay,
                 dim,
                 sampling_interval=1,
                 distance_type='deltacon0',
                 test_type='z-test',
                 num_bins=50):
        """
        :param name:
        :param partition_function: partition_function that will take trace values as an input and returns 0 or 1
        :param time_delay: delayed time interval used in phase space reconstruction
        :param dim: the dimension of embedding delayed time series (vectors)
        :param sampling_interval: used to sample the delayed time series, default is 1
        :param distance_type: 'edit_distance', 'vertex_edge_overlap', 'vertex_edge_distance', 'lambda_dist', 'netsimile',
        'resistance_distance', 'deltacon0'
        :param test_type: 'z-test', 't-test', 'chi-test', 'ks_2samp', 'cramervonmises_2samp'
        :param number of hist bins for chi2 test
        see in https://github.com/peterewills/NetComp/blob/master/netcomp/distance/exact.py
        """
        self.time_delay = time_delay
        self.dim = dim
        self.sampling_interval = sampling_interval
        self.distance_type = distance_type
        self.test_type = test_type
        self.num_bins = num_bins
        PartitionerEngine.__init__(self, name, partition_function, range(2), None)
        self.logger.debug('Creating GraphTestEngine  "%s". ' % name)

    def _initialize(self):

        self._samples_by_partition = [None] * self._partition_size
        self._partition_count = np.zeros((self._partition_size,), dtype=np.double)
        self._batch_count = 0

        self.size_in_memory += self._partition_count.nbytes

    def _update(self, batch):
        partition_values = list(map(self._partition_function, batch.values))
        for i, v in enumerate(partition_values):
            idx = self._partition_range_to_index[v]
            self._partition_count[idx] += 1
            one_leakage = np.array(batch.leakages[i], ndmin=2)
            self._samples_by_partition[idx] = one_leakage if not isinstance(self._samples_by_partition[idx],
                                                                            np.ndarray) else \
                np.concatenate((self._samples_by_partition[idx], one_leakage), axis=0)

    def _finalize(self):
        random_set, fixed_set = self._samples_by_partition[0], self._samples_by_partition[1]
        m_r, m_f = int(self._partition_count[0]), int(self._partition_count[1])

        # convert 1-d time series into 2-d graphs by phase space reconstruction
        init_graph_r = Phase_Space_Reconstruction_Graph(random_set, self.time_delay, self.dim, self.sampling_interval)
        init_graph_r.generate()
        init_graph_f = Phase_Space_Reconstruction_Graph(fixed_set, self.time_delay, self.dim, self.sampling_interval)
        init_graph_f.generate()

        self.number_of_nodes = init_graph_r.number_of_nodes

        # form a rdpg (distribution)
        # grdpg_r = Generalised_RDPG(init_graph_r.adj_matrix).generate()
        # grdpg_f = Generalised_RDPG(init_graph_f.adj_matrix).generate()
        random_sample = init_graph_r.adj_matrix
        fixed_sample = init_graph_f.adj_matrix
        random_sample_copy = np.copy(random_sample)
        np.random.shuffle(random_sample_copy)

        self.sample_size = min(m_r, m_f)

        d0, d1 = np.zeros(self.sample_size), np.zeros(self.sample_size)
        for i in tqdm(range(self.sample_size)):
            d0[i] = eval('nc.' + self.distance_type)(random_sample[i], random_sample_copy[i])
            d1[i] = eval('nc.' + self.distance_type)(random_sample[i], fixed_sample[i])

        # m0, sigma0 = np.mean(d0), np.std(d0)
        # distance_contrast = (d1 - m0) / sigma0
        # ref = (d0 - m0) / sigma0

        p_value = -1.0
        if self.test_type == 'z-test':
            z_score, p_value_z = ztest(d0, d1, value=0)
            p_value = p_value_z
        elif self.test_type == 't-test':
            t_score, p_value_t = ttest_ind(d0, d1)
            p_value = p_value_t
        elif self.test_type == 'chi-test':
            min_b, max_b = min(np.min(d0), np.min(d1)), max(np.max(d0), np.max(d1))
            hist_d0, _ = np.histogram(d0, bins=self.num_bins, range=(min_b, max_b))
            hist_d1, _ = np.histogram(d1, bins=self.num_bins, range=(min_b, max_b))
            hist_d0, hist_d1 = np.array(hist_d0, ndmin=2), np.array(hist_d1, ndmin=2)

            chi2_table = np.concatenate((hist_d0, hist_d1), axis=0)
            p_value = my_chi2test(chi2_table)
            if p_value == 0:
                # minimum value in float
                p_value = np.finfo(float).tiny
        elif self.test_type == 'ks_2samp':
            ks2_score, p_value_ks2 = ks_2samp(d0, d1)
            p_value = p_value_ks2
        elif self.test_type == 'cramervonmises_2samp':
            cm2_score, p_value_cm2 = cramervonmises_2samp(d0, d1).statistic, cramervonmises_2samp(d0, d1).pvalue
            p_value = p_value_cm2

        return p_value

    def _clean(self):
        del self._samples_by_partition
        del self._partition_count
        gc.collect()
        self.size_in_memory = 0


class GraphDistanceEngine_Attack(GuessEngine):
    """
        attack version of GraphDistanceEngine, pls ref to the DpaEngine
        different key guesses can be used hypothetical testing used in this engine is chi2 testing (ref to the Chi2Engine)
    """

    def __init__(self,
                 name,
                 selection_function,
                 guess_range,
                 time_delay,
                 dim,
                 sampling_interval=1,
                 distance_type='deltacon0',
                 num_bins=50,
                 solution=-1):
        """
        :param name:
        :param param selection_function: takes a value and a guess_guess as input, returns 0 or 1.
        :param guess_range: what are the values for the guess guess
        :param time_delay: delayed time interval used in phase space reconstruction
        :param dim: the dimension of embedding delayed time series (vectors)
        :param sampling_interval: used to sample the delayed time series, default is 1
        :param distance_type: 'edit_distance', 'vertex_edge_overlap', 'vertex_edge_distance', 'lambda_dist', 'netsimile',
        'resistance_distance', 'deltacon0'
        :param solution: if known, indicate the correct guess guess.
        """
        self.time_delay = time_delay
        self.dim = dim
        self.sampling_interval = sampling_interval
        self.distance_type = distance_type
        self.num_bins = num_bins
        self.solution = solution
        GuessEngine.__init__(self, name, selection_function, guess_range, solution)
        self.output_parser_mode = "max"
        self.logger.debug(
            'Creating GraphDistanceEngine_Attack "%s" with %d guesses.', name, len(guess_range)
        )

    def _initialize(self):
        self._samples_by_selection = [[None] * 2 for _ in range(self._number_of_guesses)]
        self._count_x = np.zeros((self._number_of_guesses, 2,), np.double)
        self._test_results = np.zeros((1, self._number_of_guesses), np.double)

    def _update(self, batch):
        for i in range(len(batch)):
            y = np.array(
                [self._function(batch.values[i], guess) for guess in self._guess_range]
            )

            idx_0 = np.where(y == 0)[0]
            idx_1 = np.where(y == 1)[0]

            leakage = np.array(batch.leakages[i], ndmin=2)
            for idx_0_i in idx_0:
                self._samples_by_selection[idx_0_i][0] = leakage if not isinstance(
                    self._samples_by_selection[idx_0_i][0], np.ndarray) \
                    else np.concatenate((self._samples_by_selection[idx_0_i][0], leakage), axis=0)
            self._count_x[idx_0, 0] += 1

            for idx_1_i in idx_1:
                self._samples_by_selection[idx_1_i][1] = leakage if not isinstance(
                    self._samples_by_selection[idx_1_i][1], np.ndarray) \
                    else np.concatenate((self._samples_by_selection[idx_1_i][1], leakage), axis=0)
            self._count_x[idx_1, 1] += 1

    def _finalize(self):
        for guess in tqdm(range(self._number_of_guesses)):
            sample0, sample1 = self._samples_by_selection[guess][0], self._samples_by_selection[guess][1]
            m_r, m_f = int(self._count_x[guess][0]), int(self._count_x[guess][1])

            # convert 1-d time series into 2-d graphs by phase space reconstruction
            init_graph0 = Phase_Space_Reconstruction_Graph(sample0, self.time_delay, self.dim, self.sampling_interval)
            init_graph0.generate()
            init_graph1 = Phase_Space_Reconstruction_Graph(sample1, self.time_delay, self.dim, self.sampling_interval)
            init_graph1.generate()

            self.number_of_nodes = init_graph0.number_of_nodes

            graph_sample0 = init_graph0.adj_matrix
            graph_sample1 = init_graph1.adj_matrix
            graph_sample0_copy = np.copy(graph_sample0)
            np.random.shuffle(graph_sample0_copy)

            self.sample_size = min(m_r, m_f)

            d0, d1 = np.zeros(self.sample_size), np.zeros(self.sample_size)
            for i in range(self.sample_size):
                d0[i] = eval('nc.' + self.distance_type)(graph_sample0[i], graph_sample0_copy[i])
                d1[i] = eval('nc.' + self.distance_type)(graph_sample0[i], graph_sample1[i])

            min_b, max_b = min(np.min(d0), np.min(d1)), max(np.max(d0), np.max(d1))
            hist_d0, _ = np.histogram(d0, bins=self.num_bins, range=(min_b, max_b))
            hist_d1, _ = np.histogram(d1, bins=self.num_bins, range=(min_b, max_b))
            hist_d0, hist_d1 = np.array(hist_d0, ndmin=2), np.array(hist_d1, ndmin=2)

            chi2_table = np.concatenate((hist_d0, hist_d1), axis=0)
            p_value = my_chi2test(chi2_table)
            if p_value == 0:
                # minimum value in float
                p_value = np.finfo(float).tiny
            self._test_results[0, guess] = p_value
        return self._test_results

    @staticmethod
    def histogram_count(data, b_arr):
        hist_count = np.zeros(b_arr.shape[0] - 1)
        for data_i in data:
            index = bisect_left(b_arr, data_i)
            if not index == 0:
                index -= 1
            hist_count[index] += 1
        return hist_count


def my_chi2test(chi2_table):
    n = np.sum(chi2_table)
    col_sum = np.sum(chi2_table, axis=0, keepdims=True)
    row_sum = np.sum(chi2_table, axis=1, keepdims=True)
    expected_freq = np.dot(row_sum, col_sum) / n
    tmp1 = (chi2_table - expected_freq) ** 2
    tmp2 = np.divide(tmp1, expected_freq, out=np.zeros_like(tmp1), where=expected_freq != 0)
    chi_score = np.sum(tmp2)
    dof = (chi2_table.shape[0] - 1) * (chi2_table.shape[1] - 1)
    p_value = chi2.sf(chi_score, dof)
    return p_value
