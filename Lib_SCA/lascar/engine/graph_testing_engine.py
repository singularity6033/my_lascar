from tqdm import tqdm
import numpy as np
import netcomp as nc

from . import T_GraphConstructionPSR, T_GraphConstructionAB, T_GraphConstructionTraceBatch
from . import T_GraphConstructionTraceAllCorr, T_GraphConstructionTraceAllDist
from .graph_testing_basic import TwoSamplesGraphTesting, GraphCommunityBasedTesting, Chi2Test
from .graph_construction_basic import BaseGraph


class GraphTestEnginePSR(T_GraphConstructionPSR):

    def __init__(self,
                 name,
                 partition_function,
                 mode='direct',
                 time_delay=2,
                 dim=3,
                 sampling_interval=1,
                 type='corr',
                 optimal=False,
                 to_unweighted=False,
                 r=3,
                 num_bins=50
                 ):
        """
        :param name:
        :param partition_function: partition_function that will take trace values as an input and returns 0 or 1
        :param mode: 'direct' --> directly apply two-samples graph testing; apply graph distance measurements
                     (Wills, Peter, and François G. Meyer. "Metrics for graph comparison: a practitioner’s guide."
                     Plos one 15.2 (2020): e0228728. https://github.com/peterewills/NetComp/blob/master/netcomp/distance/exact.py)
                     first then do chi2testing --> 'edit_distance', 'vertex_edge_overlap', 'vertex_edge_distance',
                                                   'lambda_dist', 'netsimile', 'resistance_distance', 'deltacon0'
        :param time_delay: delayed time interval used in phase space reconstruction
        :param dim: the dimension of embedding delayed time series (vectors)
        :param sampling_interval: used to sample the delayed time series, default is 1
        :param type: different measurements to calculate connectivity among nodes (embedded vectors); 'l2' -> 2-norm,
                     'corr' -> pearson correlation coefficient
        :param optimal: False -> use pre-defined delayed time step and embedded dimension in PSRG; True -> use c-c method
                        and fnn to determine the best value for delayed time step and embedded dimension automatically
        :param to_unweighted: save thresholds or not
        :param r: the number of communities (or rank) r only for approximation of P and Q used in the Tracy-Widom test
                  noted that the power of the test is not sensitive to the choice of r
        :param num_bins: number of hist bins for chi2 test
        """
        self.mode = mode
        self.type = type
        self.optimal = optimal
        self.num_bins = num_bins

        self.GraphTestingMethod = TwoSamplesGraphTesting(r)

        T_GraphConstructionPSR.__init__(self, name, partition_function, time_delay, dim, sampling_interval,
                                        type, optimal, to_unweighted)

    def _finalize(self):
        p_value = -1
        m_r, m_f = self._partition_count[0], self._partition_count[1]
        sample_size = (m_r + m_f) // 2
        rg, fg = self.adj_matrix[0], self.adj_matrix[1]
        number_of_nodes = rg[0].shape[0]
        graph_samples_r = np.array(rg) if sample_size != 1 else rg
        graph_samples_f = np.array(fg) if sample_size != 1 else fg

        # threshold used to convert weighted graph to unweighted one
        if self.to_unweighted:
            threshold_r = np.mean(self.w_to_unw_thresholds[0])
            threshold_f = np.mean(self.w_to_unw_thresholds[1])
            threshold = (threshold_r + threshold_f) / 2

            # convert to unweighted graphs
            if self.type == 'corr':
                idx_r0, idx_r1 = graph_samples_r > threshold, graph_samples_r <= threshold
                idx_f0, idx_f1 = graph_samples_f > threshold, graph_samples_f <= threshold
                graph_samples_r[idx_r0], graph_samples_r[idx_r1] = 1, 0
                graph_samples_f[idx_f0], graph_samples_f[idx_f1] = 1, 0
            elif self.type == 'dist':
                idx_r0, idx_r1 = graph_samples_r >= threshold, graph_samples_r < threshold
                idx_f0, idx_f1 = graph_samples_f >= threshold, graph_samples_f < threshold
                graph_samples_r[idx_r0], graph_samples_r[idx_r1] = 0, 1
                graph_samples_f[idx_f0], graph_samples_f[idx_f1] = 0, 1

        if self.mode == 'direct':
            if sample_size > self.number_of_nodes:
                p_value = self.GraphTestingMethod.mLarge_Testing(graph_samples_r, graph_samples_f, m_r, m_f,
                                                                 number_of_nodes)
            elif 2 <= sample_size < self.number_of_nodes:
                p_value = self.GraphTestingMethod.mSmall_Testing(graph_samples_r, graph_samples_f, m_r, m_f,
                                                                 number_of_nodes)
            elif sample_size == 1:
                p_value = self.GraphTestingMethod.mOne_Testing(graph_samples_r, graph_samples_f, number_of_nodes)
        else:
            random_sample_r_copy = np.copy(graph_samples_r)
            np.random.shuffle(random_sample_r_copy)
            d0, d1 = np.zeros(sample_size), np.zeros(sample_size)
            for i in tqdm(range(sample_size)):
                if self.mode == 'lambda_dist':
                    d0[i] = nc.lambda_dist(graph_samples_r[i],
                                           random_sample_r_copy[i],
                                           k=self.number_of_nodes,
                                           kind='adjacency')
                    d1[i] = nc.lambda_dist(graph_samples_r[i],
                                           graph_samples_f[i],
                                           k=self.number_of_nodes,
                                           kind='adjacency')
            p_value = Chi2Test(d0, d1, self.num_bins).output()
        return p_value


class GraphTestEngineAB(T_GraphConstructionAB):

    def __init__(self,
                 name,
                 partition_function,
                 mode='direct',
                 num_of_amp_groups=10,
                 num_of_moments=4,
                 type='corr',
                 to_unweighted=False,
                 r=3,
                 num_bins=50
                 ):
        """
        :param name:
        :param partition_function: partition_function that will take trace values as an input and returns 0 or 1
        :param mode: 'direct' --> directly apply two-samples graph testing; apply graph distance measurements
                     (Wills, Peter, and François G. Meyer. "Metrics for graph comparison: a practitioner’s guide."
                     Plos one 15.2 (2020): e0228728. https://github.com/peterewills/NetComp/blob/master/netcomp/distance/exact.py)
                     first then do chi2testing --> 'edit_distance', 'vertex_edge_overlap', 'vertex_edge_distance',
                                                   'lambda_dist', 'netsimile', 'resistance_distance', 'deltacon0'
        :param num_of_amp_groups: number of divided amplitude groups (equally divided between min and max)
        :param num_of_moments: number of the first moments used to construct moments vector
        :param type: different measurements to calculate connectivity among nodes (embedded vectors); 'l2' -> 2-norm,
                     'corr' -> pearson correlation coefficient
        :param to_unweighted: save thresholds or not
        :param r: the number of communities (or rank) r only for approximation of P and Q used in the Tracy-Widom test
                  noted that the power of the test is not sensitive to the choice of r
        :param num_bins: number of hist bins for chi2 test
        """
        self.mode = mode
        self.type = type
        self.num_bins = num_bins

        self.GraphTestingMethod = TwoSamplesGraphTesting(r)

        T_GraphConstructionAB.__init__(self, name, partition_function, num_of_amp_groups, num_of_moments, type, to_unweighted)

    def _finalize(self):
        p_value = -1
        m_r, m_f = self._partition_count[0], self._partition_count[1]
        sample_size = (m_r + m_f) // 2
        rg, fg = self.adj_matrix[0], self.adj_matrix[1]
        number_of_nodes = rg[0].shape[0]
        graph_samples_r = np.array(rg) if sample_size != 1 else rg
        graph_samples_f = np.array(fg) if sample_size != 1 else fg

        # threshold used to convert weighted graph to unweighted one
        if self.to_unweighted:
            threshold_r = np.mean(self.w_to_unw_thresholds[0])
            threshold_f = np.mean(self.w_to_unw_thresholds[1])
            threshold = (threshold_r + threshold_f) / 2

            # convert to unweighted graphs
            if self.type == 'corr':
                idx_r0, idx_r1 = graph_samples_r > threshold, graph_samples_r <= threshold
                idx_f0, idx_f1 = graph_samples_f > threshold, graph_samples_f <= threshold
                graph_samples_r[idx_r0], graph_samples_r[idx_r1] = 1, 0
                graph_samples_f[idx_f0], graph_samples_f[idx_f1] = 1, 0
            elif self.type == 'dist':
                idx_r0, idx_r1 = graph_samples_r >= threshold, graph_samples_r < threshold
                idx_f0, idx_f1 = graph_samples_f >= threshold, graph_samples_f < threshold
                graph_samples_r[idx_r0], graph_samples_r[idx_r1] = 0, 1
                graph_samples_f[idx_f0], graph_samples_f[idx_f1] = 0, 1

        if self.mode == 'direct':
            if sample_size > self.number_of_nodes:
                p_value = self.GraphTestingMethod.mLarge_Testing(graph_samples_r, graph_samples_f, m_r, m_f,
                                                                 number_of_nodes)
            elif 2 <= sample_size < self.number_of_nodes:
                p_value = self.GraphTestingMethod.mSmall_Testing(graph_samples_r, graph_samples_f, m_r, m_f,
                                                                 number_of_nodes)
            elif sample_size == 1:
                p_value = self.GraphTestingMethod.mOne_Testing(graph_samples_r, graph_samples_f, number_of_nodes)

        else:
            random_sample_r_copy = np.copy(graph_samples_r)
            np.random.shuffle(random_sample_r_copy)
            d0, d1 = np.zeros(sample_size), np.zeros(sample_size)
            for i in tqdm(range(sample_size)):
                if self.mode == 'lambda_dist':
                    d0[i] = nc.lambda_dist(graph_samples_r[i],
                                           random_sample_r_copy[i],
                                           k=self.number_of_nodes,
                                           kind='adjacency')
                    d1[i] = nc.lambda_dist(graph_samples_r[i],
                                           graph_samples_f[i],
                                           k=self.number_of_nodes,
                                           kind='adjacency')
            p_value = Chi2Test(d0, d1, self.num_bins).output()
        return p_value


class GraphTestEngineTraceBatch(T_GraphConstructionTraceBatch):
    """
    correlation version
    """

    def __init__(self, name, partition_function, mode='direct', to_unweighted=False, r=3, num_bins=50):
        """
        :param name:
        :param partition_function: partition_function that will take trace values as an input and returns 0 or 1
        :param mode: 'direct' --> directly apply two-samples graph testing; apply graph distance measurements
                     (Wills, Peter, and François G. Meyer. "Metrics for graph comparison: a practitioner’s guide."
                     Plos one 15.2 (2020): e0228728. https://github.com/peterewills/NetComp/blob/master/netcomp/distance/exact.py)
                     first then do chi2testing --> 'edit_distance', 'vertex_edge_overlap', 'vertex_edge_distance',
                                                   'lambda_dist', 'netsimile', 'resistance_distance', 'deltacon0'
        :param to_unweighted: save thresholds or not
        :param r: the number of communities (or rank) r only for approximation of P and Q used in the Tracy-Widom test
                  noted that the power of the test is not sensitive to the choice of r
        :param num_bins: number of hist bins for chi2 test
        """
        self.mode = mode
        self.num_bins = num_bins
        self.GraphTestingMethod = TwoSamplesGraphTesting(r)

        T_GraphConstructionTraceBatch.__init__(self, name, partition_function, to_unweighted)

    def _finalize(self):
        p_value = -1
        m_r, m_f = self._partition_count[0], self._partition_count[1]
        sample_size = (m_r + m_f) // 2
        rg, fg = self.adj_matrix[0], self.adj_matrix[1]
        number_of_nodes = rg[0].shape[0]
        graph_samples_r = np.array(rg) if sample_size != 1 else rg
        graph_samples_f = np.array(fg) if sample_size != 1 else fg

        # threshold used to convert weighted graph to unweighted one
        if self.to_unweighted:
            threshold_r = np.mean(self.w_to_unw_thresholds[0])
            threshold_f = np.mean(self.w_to_unw_thresholds[1])
            threshold = (threshold_r + threshold_f) / 2

            # convert to unweighted graphs
            idx_r0, idx_r1 = graph_samples_r > threshold, graph_samples_r <= threshold
            idx_f0, idx_f1 = graph_samples_f > threshold, graph_samples_f <= threshold
            graph_samples_r[idx_r0], graph_samples_r[idx_r1] = 1, 0
            graph_samples_f[idx_f0], graph_samples_f[idx_f1] = 1, 0

        if self.mode == 'direct':
            if sample_size > self.number_of_nodes:
                p_value = self.GraphTestingMethod.mLarge_Testing(graph_samples_r, graph_samples_f, m_r, m_f,
                                                                 number_of_nodes)
            elif 2 <= sample_size < self.number_of_nodes:
                p_value = self.GraphTestingMethod.mSmall_Testing(graph_samples_r, graph_samples_f, m_r, m_f,
                                                                 number_of_nodes)
            elif sample_size == 1:
                p_value = self.GraphTestingMethod.mOne_Testing(graph_samples_r, graph_samples_f, number_of_nodes)
        else:
            random_sample_r_copy = np.copy(graph_samples_r)
            np.random.shuffle(random_sample_r_copy)
            d0, d1 = np.zeros(sample_size), np.zeros(sample_size)
            for i in tqdm(range(sample_size)):
                if self.mode == 'lambda_dist':
                    d0[i] = nc.lambda_dist(graph_samples_r[i],
                                           random_sample_r_copy[i],
                                           k=self.number_of_nodes,
                                           kind='adjacency')
                    d1[i] = nc.lambda_dist(graph_samples_r[i],
                                           graph_samples_f[i],
                                           k=self.number_of_nodes,
                                           kind='adjacency')
            p_value = Chi2Test(d0, d1, self.num_bins).output()
        return p_value


class GraphTestingEngineTraceAllCorr(T_GraphConstructionTraceAllCorr):
    """
    convert all traces in to one graph, each column (along trace axis) represents one node
    using pearson correlation coefficient to calculate connectivity among nodes (embedded vectors)
    """

    def __init__(self,
                 name,
                 partition_function,
                 to_unweighted=False,
                 k=3,
                 ):
        """
        :param name:
        :param partition_function: partition_function that will take trace values as an input and returns 0 or 1
        :param to_unweighted: save thresholds or not
        :param k: the number of communities (or rank) r only for approximation of P and Q used in the Tracy-Widom test
                  noted that the power of the test is not sensitive to the choice of r

        """
        self.to_unweighted = to_unweighted
        self.k = k

        self.bg = BaseGraph()
        self.GraphTestingMethod = TwoSamplesGraphTesting(k)
        T_GraphConstructionTraceAllCorr.__init__(self, name, partition_function)

    def _finalize(self):
        size_r, size_f = self._partition_count[0], self._partition_count[1]
        self.l[0], self.l[1] = self.l[0] / size_r, self.l[1] / size_f
        self.l2[0], self.l2[1] = self.l2[0] / size_r, self.l2[1] / size_f
        self.ll[0], self.ll[1] = self.ll[0] / size_r, self.ll[1] / size_f

        v_r, v_f = self.l2[0] - self.l[0] ** 2, self.l2[1] - self.l[1] ** 2
        numerator_r, numerator_f = self.ll[0] - np.outer(self.l[0], self.l[0]), self.ll[1] - np.outer(self.l[1],
                                                                                                      self.l[1])
        denominator_r, denominator_f = np.sqrt(np.outer(v_r, v_r)), np.sqrt(np.outer(v_f, v_f))
        mask_r, mask_f = v_r == 0.0, v_f == 0.0
        numerator_r[mask_r, mask_r], denominator_r[mask_r, mask_r] = 0.0, 1.0
        numerator_f[mask_f, mask_f], denominator_f[mask_f, mask_f] = 0.0, 1.0

        graph_samples_r = np.abs(np.nan_to_num(numerator_r / denominator_r))
        graph_samples_f = np.abs(np.nan_to_num(numerator_f / denominator_f))

        if self.to_unweighted:
            threshold_r = self.bg.calc_best_conversion_threshold(graph_samples_r, self.number_of_nodes)
            threshold_f = self.bg.calc_best_conversion_threshold(graph_samples_f, self.number_of_nodes)
            threshold = (threshold_r + threshold_f) / 2

            # convert to unweighted graphs
            idx_r0, idx_r1 = graph_samples_r > threshold, graph_samples_r <= threshold
            idx_f0, idx_f1 = graph_samples_f > threshold, graph_samples_f <= threshold
            graph_samples_r[idx_r0], graph_samples_r[idx_r1] = 1, 0
            graph_samples_f[idx_f0], graph_samples_f[idx_f1] = 1, 0

        # p_value = self.GraphTestingMethod.mOne_Testing(graph_samples_r, graph_samples_f, self.number_of_nodes)
        ts_ct = GraphCommunityBasedTesting(graph_samples_r, graph_samples_f, self.number_of_nodes, self.k)
        p_value = ts_ct.two_samples_community_memberships_testing()

        return p_value


class GraphTestingEngineTraceAllDist(T_GraphConstructionTraceAllDist):
    """
    convert all traces in to one graph, each column (along trace axis) represents one node
    using Kolmogorov–Smirnov (K-S) statistics to calculate connectivity among nodes (embedded vectors)
    """

    def __init__(self,
                 name,
                 partition_function,
                 to_unweighted=False,
                 num_bins=100,
                 k=3,
                 ):
        """
        :param name:
        :param partition_function: partition_function that will take trace values as an input and returns 0 or 1
        :param k: the number of communities (or rank) r only for approximation of P and Q used in the Tracy-Widom test
                  noted that the power of the test is not sensitive to the choice of r
        :param num_bins: used for estimation of cdf by histogram
        """
        self.to_unweighted = to_unweighted
        self.num_bins = num_bins
        self.k = k
        self.GraphTestingMethod = TwoSamplesGraphTesting(k)

        T_GraphConstructionTraceAllDist.__init__(self, name, partition_function, num_bins)
        self.logger.debug('Creating GraphTestEngine  "%s". ' % name)

    def _finalize(self):
        m_r, m_f = self._partition_count[0], self._partition_count[1]
        pdf_r, pdf_f = np.array(self.hist_counts[0]) / m_r, np.array(self.hist_counts[1]) / m_f
        pdf_r_with_matrix = np.reshape(pdf_r, (-1, self.number_of_nodes), order='F')
        pdf_f_with_matrix = np.reshape(pdf_f, (-1, self.number_of_nodes), order='F')
        cdf_r_with_matrix = np.cumsum(pdf_r_with_matrix, axis=0)
        cdf_f_with_matrix = np.cumsum(pdf_f_with_matrix, axis=0)

        # more large the weight is, more close two nodes are
        graph_samples_r = self._calc_kolmogorov_smirnov_distance(cdf_r_with_matrix)
        graph_samples_f = self._calc_kolmogorov_smirnov_distance(cdf_f_with_matrix)

        if self.to_unweighted:
            threshold_r = self.BaseGraph.calc_best_conversion_threshold(graph_samples_r, self.number_of_nodes)
            threshold_f = self.BaseGraph.calc_best_conversion_threshold(graph_samples_f, self.number_of_nodes)

            threshold = (threshold_r + threshold_f) / 2

            idx_r0, idx_r1 = graph_samples_r >= threshold, graph_samples_r < threshold
            idx_f0, idx_f1 = graph_samples_f >= threshold, graph_samples_f < threshold
            graph_samples_r[idx_r0], graph_samples_r[idx_r1] = 0, 1
            graph_samples_f[idx_f0], graph_samples_f[idx_f1] = 0, 1

        ts_ct = GraphCommunityBasedTesting(graph_samples_r, graph_samples_r, self.number_of_nodes, self.k)
        p_value = ts_ct.two_samples_community_memberships_testing()
        # p_value = self.GraphTestingMethod.mOne_Testing(graph_samples_r, graph_samples_f, self.number_of_nodes)

        return p_value

    def _calc_kolmogorov_smirnov_distance(self, cdf_with_matrix):
        # try to construct to #nodes * #nodes size of column vectors to speedup the calculation
        tmp1 = np.repeat(cdf_with_matrix, self.number_of_nodes, axis=1)
        tmp2 = np.tile(cdf_with_matrix, self.number_of_nodes)
        cdf_diff = np.abs(tmp1 - tmp2)
        ks_dist = np.max(cdf_diff, axis=0)  # size is #nodes * #nodes
        adj_matrix = np.reshape(ks_dist, (self.number_of_nodes, self.number_of_nodes), order='C')
        return adj_matrix

    # def _kolmogorov_smirnov_distance(self, data1, data2):
    #     adj_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes))
    #     data1 = np.sort(data1, axis=0)
    #     data2 = np.sort(data2, axis=0)
    #     n1 = data1.shape[0]
    #     n2 = data2.shape[0]
    #     if min(n1, n2) == 0:
    #         raise ValueError('Data passed to _kolmogorov_smirnov_distance must not be empty')
    #     print('[INFO] calculating KS distance...')
    #     for i in tqdm(range(self.number_of_nodes)):
    #         for j in range(i + 1, self.number_of_nodes):
    #             dat1, dat2 = data1[:, i].flatten(), data2[:, j].flatten()
    #             data_all = np.concatenate([dat1, dat2])
    #             # using searchsorted solves equal data problem
    #             cdf1 = np.searchsorted(dat1, data_all, side='right') / n1
    #             cdf2 = np.searchsorted(dat2, data_all, side='right') / n2
    #             cdf_diffs = cdf1 - cdf2
    #
    #             # Identify the location of the statistic
    #             min_idx = np.argmin(cdf_diffs)
    #             max_idx = np.argmax(cdf_diffs)
    #
    #             # Ensure sign of minS is not negative.
    #             minS = np.clip(-cdf_diffs[min_idx], 0, 1)
    #             maxS = cdf_diffs[max_idx]
    #             adj_matrix[i, j] = maxS if maxS >= minS else minS
    #             adj_matrix[j, i] = adj_matrix[i, j]
    #     return adj_matrix
