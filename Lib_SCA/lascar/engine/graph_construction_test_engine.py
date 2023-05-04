"""
GraphTestEngine is a two-samples graph testing used to compute whether two graph samples G and H are drawn from
the same distribution or not
"""
from collections import defaultdict
import numpy as np

from . import PartitionerEngine
from .graph_construction_basic import BaseGraph, PhaseSpaceReconstructionGraph, AmplitudesBasedGraph


### one trace converted into one graph ###
class T_GraphConstructionPSR(PartitionerEngine):
    """
    testing version of PSR graph --> see details in PhaseSpaceReconstructionGraph
    """

    def __init__(self,
                 name,
                 partition_function,
                 time_delay=2,
                 dim=3,
                 sampling_interval=1,
                 type='corr',
                 optimal=False,
                 to_unweighted=False,
                 ):
        self.to_unweighted = to_unweighted
        self.psr_g = PhaseSpaceReconstructionGraph(time_delay, dim, sampling_interval, type, optimal, to_unweighted)

        PartitionerEngine.__init__(self, name, partition_function, range(2), None)
        self.logger.debug('Creating T_GraphConstructionPSR  "%s". ' % name)

    def _initialize(self):
        self.adj_matrix, self.w_to_unw_thresholds = defaultdict(list), defaultdict(list)
        self.number_of_time_points = self._session.leakage_shape[0]
        self._partition_count = np.zeros((self._partition_size,), dtype=np.double)

    def _update(self, batch):
        partition_values = np.array(list(map(self._partition_function, batch.values)))
        for i, v in enumerate(partition_values):
            idx = self._partition_range_to_index[v]
            self._partition_count[idx] += 1
            g, threshold = self.psr_g.generate(batch.leakages[i])
            self.adj_matrix[idx].append(g)
            if threshold:
                self.w_to_unw_thresholds[idx].append(threshold)


class T_GraphConstructionAB(PartitionerEngine):
    """
    testing version of AB graph --> see details in AmplitudesBasedGraph
    """

    def __init__(self,
                 name,
                 partition_function,
                 num_of_amp_groups=10,
                 num_of_moments=4,
                 type='corr',
                 to_unweighted=False,
                 ):

        self.num_of_amp_groups = num_of_amp_groups
        self.num_of_moments = num_of_moments
        self.type = type
        self.to_unweighted = to_unweighted

        self.ab_g = AmplitudesBasedGraph(num_of_amp_groups, num_of_moments, type, to_unweighted)

        PartitionerEngine.__init__(self, name, partition_function, range(2), None)
        self.logger.debug('Creating T_GraphConstructionAB  "%s". ' % name)

    def _initialize(self):
        self.adj_matrix, self.w_to_unw_thresholds = defaultdict(list), defaultdict(list)
        self.number_of_time_points = self._session.leakage_shape[0]
        self._partition_count = np.zeros((self._partition_size,), dtype=np.double)

    def _update(self, batch):
        partition_values = np.array(list(map(self._partition_function, batch.values)))
        for i, v in enumerate(partition_values):
            idx = self._partition_range_to_index[v]
            self._partition_count[idx] += 1
            g, threshold = self.ab_g.generate(batch.leakages[i])
            self.adj_matrix[idx].append(g)
            if threshold:
                self.w_to_unw_thresholds[idx].append(threshold)


### multiple traces converted into one graph ###
class T_GraphConstructionTraceBatch(PartitionerEngine):

    def __init__(self, name, partition_function, to_unweighted=False):
        """
        :param partition_function: partition_function that will take trace values as an input and returns 0 or 1
        :param to_unweighted: save thresholds or not
        """
        self.BaseGraph = BaseGraph()
        self.to_unweighted = to_unweighted

        PartitionerEngine.__init__(self, name, partition_function, range(2), None)
        self.logger.debug('Creating T_GraphConstructionTraceBatch  "%s". ' % name)

    def _initialize(self):
        self.adj_matrix, self.w_to_unw_thresholds = defaultdict(list), defaultdict(list)
        self.number_of_time_points = self._session.leakage_shape[0]
        self._partition_count = np.zeros((self._partition_size,), dtype=np.double)

    def _update(self, batch):
        self._samples_by_partition = defaultdict(list)
        self._partition_count = np.zeros((self._partition_size,), dtype=np.double)

        partition_values = np.array(list(map(self._partition_function, batch.values)))
        for pi in range(self._partition_size):
            lk = batch.leakages[partition_values == pi]
            self._partition_count[pi] = lk.shape[0]
            self._samples_by_partition[pi] = lk

        set0, set1 = self._samples_by_partition[0], self._samples_by_partition[1]
        self.number_of_nodes = int((self._partition_count[0] + self._partition_count[1]) // 2)

        # convert multiple traces to one graph
        adj_matrix0 = self.BaseGraph.calc_correlation_matrix(set0, set0)
        adj_matrix1 = self.BaseGraph.calc_correlation_matrix(set1, set1)
        self.adj_matrix[0].append(adj_matrix0)
        self.adj_matrix[1].append(adj_matrix1)

        # record threshold info
        if self.to_unweighted:
            self.w_to_unw_thresholds[0].append(self.BaseGraph.calc_best_conversion_threshold(adj_matrix0, self.number_of_nodes))
            self.w_to_unw_thresholds[1].append(self.BaseGraph.calc_best_conversion_threshold(adj_matrix1, self.number_of_nodes))


### all traces converted into one graph ###
class T_GraphConstructionTraceAllCorr(PartitionerEngine):
    """
    convert all traces in to one graph, each column (along trace axis) represents one node
    using pearson correlation coefficient to calculate connectivity among nodes (embedded vectors)
    """

    def __init__(self,
                 name,
                 partition_function,
                 ):
        """
        :param name:
        :param partition_function: partition_function that will take trace values as an input and returns 0 or 1
        """

        PartitionerEngine.__init__(self, name, partition_function, range(2), None)
        self.logger.debug('T_GraphConstructionTraceAllCorr "%s". ' % name)

    def _initialize(self):
        self.number_of_nodes = self._session.leakage_shape[0]
        self.l = np.zeros((self._partition_size, self.number_of_nodes), np.double)
        self.l2 = np.zeros((self._partition_size, self.number_of_nodes), np.double)
        self.ll = np.zeros((self._partition_size, self.number_of_nodes, self.number_of_nodes), np.double)
        self._partition_count = np.zeros((self._partition_size,), dtype=np.double)

    def _update(self, batch):
        partition_values = np.array(list(map(self._partition_function, batch.values)))
        leakages = batch.leakages
        for i in range(self._partition_size):
            lk = leakages[partition_values == i]
            self.l[i] += lk.sum(0)
            self.l2[i] += (lk ** 2).sum(0)
            self.ll[i] += np.dot(lk.T, lk)
            self._partition_count[i] += lk.shape[0]


class T_GraphConstructionTraceAllDist(PartitionerEngine):
    """
    convert all traces in to one graph, each column (along trace axis) represents one node
    using Kolmogorov–Smirnov (K-S) statistics to calculate connectivity among nodes (embedded vectors)
    """

    def __init__(self,
                 name,
                 partition_function,
                 num_bins=100,
                 ):
        """
        :param name:
        :param partition_function: partition_function that will take trace values as an input and returns 0 or 1
        :param num_bins: used for estimation of cdf by histogram
        """
        self.num_bins = num_bins
        self.BaseGraph = BaseGraph()

        PartitionerEngine.__init__(self, name, partition_function, range(2), None)
        self.logger.debug('T_GraphConstructionTraceAllDist "%s". ' % name)

    def _calc_histogram(self, idx, lk, min_data, max_data):
        """
        this update_hist function directly update the previous histogram based on the current data
        it may involve padding operations
        """
        sample_size = lk.shape[0]
        self._partition_count[idx] += sample_size
        lk_flat = lk.flatten(order='F')
        self._update_ranges_counts(idx, min_data, max_data)
        counts_index = np.searchsorted(self.hist_ranges, lk_flat, side='left')
        counts_index[lk_flat != self.hist_ranges[0]] -= 1

        print('[INFO] calculating histogram...')
        time_sample_idx = 0
        num_bins = len(self.hist_ranges)
        for i, c_idx in enumerate(counts_index):
            if i > 0 and i % sample_size == 0:
                time_sample_idx += 1
            self.hist_counts[idx][c_idx + time_sample_idx * num_bins] += 1

    def _update_ranges_counts(self, idx, min_data, max_data):
        if not isinstance(self.hist_counts[idx], np.ndarray):
            self.bin_width = (max_data - min_data) / self.num_bins
            self.hist_ranges = [min_data + i * self.bin_width for i in range(self.num_bins)]
            one_hist = [0] * len(self.hist_ranges)
            self.hist_counts[idx] = one_hist * self.number_of_nodes
        else:
            min_boundary, max_boundary = np.min(self.hist_ranges), np.max(self.hist_ranges)
            left_pad, right_pad = list(), list()
            while min_data < min_boundary:
                min_boundary = min_boundary - self.bin_width
                left_pad.append(min_boundary)
            while max_data - self.bin_width > max_boundary:
                max_boundary = max_boundary + self.bin_width
                right_pad.append(max_boundary)
            self.hist_ranges = left_pad[::-1] + self.hist_ranges + right_pad
            old_hist_counts = np.array(self.hist_counts[idx])
            to_mat = np.reshape(old_hist_counts, (-1, self.number_of_nodes), order='F')
            updated_hist_counts = np.pad(to_mat, ((len(left_pad), len(right_pad)), (0, 0)), 'constant',
                                         constant_values=(0, 0))
            self.hist_counts[idx] = updated_hist_counts.flatten(order='F')

    def _initialize(self):
        self.number_of_nodes = self._session.leakage_shape[0]
        self.hist_ranges = None
        self.hist_counts = [None] * self._partition_size
        self._partition_count = np.zeros((self._partition_size, 1))

    def _update(self, batch):
        partition_values = np.array(list(map(self._partition_function, batch.values)))
        leakages = batch.leakages
        min_leak, max_leak = np.min(leakages), np.max(leakages)

        for i in range(self._partition_size):
            lk = leakages[partition_values == i]
            self._calc_histogram(i, lk, min_leak, max_leak)
