import gc
from bisect import bisect_left

from tqdm import tqdm
import numpy as np
from scipy.stats import chi2, pearsonr
import networkx as nx
import netcomp as nc

from . import PartitionerEngine, GuessEngine, Phase_Space_Reconstruction_Graph


class GraphDistanceEngine(PartitionerEngine):
    """
        GraphDistanceEngine is used to calculate graph differences by some distance measurements
        and corresponding statistical testing procedure (chi2test) is also included in this engine
        based on the paper: Wills, Peter, and François G. Meyer. "Metrics for graph comparison: a practitioner’s guide." Plos one 15.2 (2020): e0228728.
    """

    def __init__(self,
                 name,
                 partition_function,
                 time_delay,
                 dim,
                 sampling_interval=1,
                 distance_type='lambda_dist',
                 num_bins=50):
        """
        :param name:
        :param partition_function: partition_function that will take trace values as an input and returns 0 or 1
        :param time_delay: delayed time interval used in phase space reconstruction
        :param dim: the dimension of embedding delayed time series (vectors)
        :param sampling_interval: used to sample the delayed time series, default is 1
        :param distance_type: 'edit_distance', 'vertex_edge_overlap', 'vertex_edge_distance', 'lambda_dist', 'netsimile',
        'resistance_distance', 'deltacon0'
        :param number of hist bins for chi2 test
        see in https://github.com/peterewills/NetComp/blob/master/netcomp/distance/exact.py
        """
        self.time_delay = time_delay
        self.dim = dim
        self.sampling_interval = sampling_interval
        self.distance_type = distance_type
        self.num_bins = num_bins
        PartitionerEngine.__init__(self, name, partition_function, range(2), None)
        self.logger.debug('Creating GraphDistanceEngine "%s". ' % name)

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

        # convert 1-d time series into 2-d weighted graphs by phase space reconstruction
        init_graph_r = Phase_Space_Reconstruction_Graph(random_set, self.time_delay, self.dim, self.sampling_interval)
        init_graph_r.generate()

        init_graph_f = Phase_Space_Reconstruction_Graph(fixed_set, self.time_delay, self.dim, self.sampling_interval)
        init_graph_f.generate()

        self.number_of_nodes = init_graph_r.number_of_nodes

        self.sample_size = min(m_r, m_f)

        random_sample = init_graph_r.adj_matrix if self.sample_size != 1 else init_graph_r.adj_matrix[0]
        fixed_sample = init_graph_f.adj_matrix if self.sample_size != 1 else init_graph_f.adj_matrix[0]

        random_sample_copy = np.copy(random_sample)
        np.random.shuffle(random_sample_copy)

        d0, d1 = np.zeros(self.sample_size), np.zeros(self.sample_size)
        for i in tqdm(range(self.sample_size)):
            if self.distance_type == 'lambda_dist':
                d0[i] = nc.lambda_dist(random_sample[i], random_sample_copy[i], k=self.number_of_nodes, kind='adjacency')
                d1[i] = nc.lambda_dist(random_sample[i], fixed_sample[i], k=self.number_of_nodes, kind='adjacency')

        # apply chi2test on two distance distribution
        min_b, max_b = min(np.min(d0), np.min(d1)), max(np.max(d0), np.max(d1))
        hist_d0, _ = np.histogram(d0, bins=self.num_bins, range=(min_b, max_b))
        hist_d1, _ = np.histogram(d1, bins=self.num_bins, range=(min_b, max_b))
        hist_d0, hist_d1 = np.array(hist_d0, ndmin=2), np.array(hist_d1, ndmin=2)

        chi2_table = np.concatenate((hist_d0, hist_d1), axis=0)
        p_value = my_chi2test(chi2_table)
        if p_value == 0:
            # minimum value in float
            p_value = np.finfo(float).tiny
        return p_value

    def _clean(self):
        del self._samples_by_partition
        del self._partition_count
        gc.collect()
        self.size_in_memory = 0


class TraceBasedGraphDistanceEngine(PartitionerEngine):
    """
        GraphDistanceEngine is used to calculate graph differences by some distance measurements
        and corresponding statistical testing procedure (chi2test) is also included in this engine
        based on the paper: Wills, Peter, and François G. Meyer. "Metrics for graph comparison: a practitioner’s guide." Plos one 15.2 (2020): e0228728.
    """

    def __init__(self,
                 name,
                 partition_function,
                 distance_type='lambda_dist',
                 num_bins=50):
        """
        :param name:
        :param partition_function: partition_function that will take trace values as an input and returns 0 or 1
        :param distance_type: 'edit_distance', 'vertex_edge_overlap', 'vertex_edge_distance', 'lambda_dist', 'netsimile',
        'resistance_distance', 'deltacon0'
        :param number of hist bins for chi2 test
        see in https://github.com/peterewills/NetComp/blob/master/netcomp/distance/exact.py
        """
        self.distance_type = distance_type
        self.num_bins = num_bins
        PartitionerEngine.__init__(self, name, partition_function, range(2), None)
        self.logger.debug('Creating GraphDistanceEngine "%s". ' % name)

    def _initialize(self):
        self._graph_set_r, self._graph_set_f = list(), list()

    def _update(self, batch):
        batch_size = batch.leakages.shape[0] // 2
        self._samples_by_partition = np.zeros((self._partition_size, batch_size,) + self._session.leakage_shape,
                                              dtype=np.double)
        self._partition_count = np.zeros((self._partition_size,), dtype=np.double)

        partition_values = list(map(self._partition_function, batch.values))
        for i, v in enumerate(partition_values):
            idx = self._partition_range_to_index[v]
            self._partition_count[idx] += 1
            # one_leakage = np.array(batch.leakages[i], ndmin=2)
            self._samples_by_partition[idx, i // 2, :] = batch.leakages[i]

        random_set, fixed_set = self._samples_by_partition[0], self._samples_by_partition[1]
        self.number_of_nodes = int((self._partition_count[0] + self._partition_count[1]) // 2)

        # convert the whole traces to one graph
        adj_matrix_r = self._calc_correlation_matrix(random_set, random_set)
        adj_matrix_f = self._calc_correlation_matrix(fixed_set, fixed_set)

        self._graph_set_r.append(adj_matrix_r)
        self._graph_set_f.append(adj_matrix_f)

    def _finalize(self):
        m_r, m_f = len(self._graph_set_r), len(self._graph_set_f)
        sample_size = (m_r + m_f) // 2

        random_sample = np.array(self._graph_set_r) if sample_size != 1 else self._graph_set_r[0]
        fixed_sample = np.array(self._graph_set_f) if sample_size != 1 else self._graph_set_f[0]
        random_sample_copy = np.copy(random_sample)
        np.random.shuffle(random_sample_copy)

        self.sample_size = min(m_r, m_f)

        d0, d1 = np.zeros(self.sample_size), np.zeros(self.sample_size)
        for i in tqdm(range(self.sample_size)):
            if self.distance_type == 'lambda_dist':
                d0[i] = nc.lambda_dist(random_sample[i], random_sample_copy[i], k=self.number_of_nodes, kind='adjacency')
                d1[i] = nc.lambda_dist(random_sample[i], fixed_sample[i], k=self.number_of_nodes, kind='adjacency')

        # apply chi2test on two distance distribution
        min_b, max_b = min(np.min(d0), np.min(d1)), max(np.max(d0), np.max(d1))
        hist_d0, _ = np.histogram(d0, bins=self.num_bins, range=(min_b, max_b))
        hist_d1, _ = np.histogram(d1, bins=self.num_bins, range=(min_b, max_b))
        hist_d0, hist_d1 = np.array(hist_d0, ndmin=2), np.array(hist_d1, ndmin=2)

        chi2_table = np.concatenate((hist_d0, hist_d1), axis=0)
        p_value = my_chi2test(chi2_table)
        if p_value == 0:
            # minimum value in float
            p_value = np.finfo(float).tiny
        return p_value

    def _clean(self):
        del self._samples_by_partition
        del self._partition_count
        gc.collect()
        self.size_in_memory = 0

    @staticmethod
    def _calc_correlation_matrix(a, b):
        num_of_vec = a.shape[0]
        res = np.zeros((num_of_vec, num_of_vec))
        for i in range(num_of_vec):
            for j in range(num_of_vec):
                if i <= j:
                    res[i, j] = pearsonr(a[i, :], b[j, :])[0]
                    res[j, i] = res[i, j]
                else:
                    continue
        return np.abs(res)


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
