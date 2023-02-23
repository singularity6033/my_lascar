import gc
from math import floor
from tqdm import tqdm
import numpy as np
from scipy.stats import norm, bernoulli, chi2, ks_2samp, cramervonmises_2samp, pearsonr
from sklearn.cluster import KMeans
from TracyWidom import TracyWidom

from . import PartitionerEngine, GuessEngine, Phase_Space_Reconstruction_Graph, Simple_PSRG


class GraphTestEngine(PartitionerEngine):
    """
    GraphTestEngine is a two-samples graph testing used to compute whether two graph samples G and H are drawn from
    the same distribution or not
    Ghoshdastidar, Debarghya, and Ulrike Von Luxburg. "Practical methods for graph two-sample testing."
    Advances in Neural Information Processing Systems 31 (2018).
    """

    def __init__(self, name, partition_function, time_delay, dim, sampling_interval=1, r=3):
        """
        :param name:
        :param partition_function: partition_function that will take trace values as an input and returns 0 or 1
        :param time_delay: delayed time interval used in phase space reconstruction
        :param dim: the dimension of embedding delayed time series (vectors)
        :param sampling_interval: used to sample the delayed time series, default is 1
        :param r: the number of communities (or rank) r only for approximation of P and Q used in the Tracy-Widom test
                  noted that the power of the test is not sensitive to the choice of r
        """
        self.time_delay = time_delay
        self.dim = dim
        self.sampling_interval = sampling_interval
        self.r = r

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
        p_value = -1.0
        random_set, fixed_set = self._samples_by_partition[0], self._samples_by_partition[1]
        m_r, m_f = self._partition_count[0], self._partition_count[1]

        # convert 1-d time series into 2-d graphs by phase space reconstruction
        # init_graph_r = Phase_Space_Reconstruction_Graph(random_set, self.time_delay, self.dim, self.sampling_interval)
        # init_graph_r.generate()
        # init_graph_f = Phase_Space_Reconstruction_Graph(fixed_set, self.time_delay, self.dim, self.sampling_interval)
        # init_graph_f.generate()

        init_graph_r = Simple_PSRG(random_set, self.time_delay, self.dim, self.sampling_interval)
        init_graph_r.generate()
        self.threshold_r = np.mean(init_graph_r.w_to_unw_thresholds)

        init_graph_f = Simple_PSRG(fixed_set, self.time_delay, self.dim, self.sampling_interval)
        init_graph_f.generate()
        self.threshold_f = np.mean(init_graph_f.w_to_unw_thresholds)

        # size of graph
        self.number_of_nodes = init_graph_r.number_of_nodes

        # convert determinate graphs into generalised RDPG which is also a type of inhomogeneous ER (IER) graphs
        # grdpg_r = Generalised_RDPG(init_graph_r.adj_matrix, self.mode).generate()
        # grdpg_f = Generalised_RDPG(init_graph_f.adj_matrix, self.mode).generate()

        sample_size = (m_r + m_f) // 2

        graph_samples_r = init_graph_r.adj_matrix if sample_size != 1 else init_graph_r.adj_matrix[0]
        graph_samples_f = init_graph_f.adj_matrix if sample_size != 1 else init_graph_f.adj_matrix[0]

        # convert to unweighted graphs
        idx_r0, idx_r1 = graph_samples_r > self.threshold_r, graph_samples_r <= self.threshold_r
        graph_samples_r[idx_r0], graph_samples_r[idx_r1] = 0, 1
        idx_f0, idx_f1 = graph_samples_f > self.threshold_f, graph_samples_f <= self.threshold_f
        graph_samples_f[idx_f0], graph_samples_f[idx_f1] = 0, 1

        if sample_size > self.number_of_nodes:
            # it is a type of chi-square test statistic
            mean_grdpg_r, mean_grdpg_f = np.mean(graph_samples_r, axis=0), np.mean(graph_samples_f, axis=0)
            var_grdpg_r, var_grdpg_f = np.var(graph_samples_r, axis=0), np.var(graph_samples_f, axis=0)

            nominator = 0
            denominator = 0
            for i in range(self.number_of_nodes):
                for j in range(self.number_of_nodes):
                    if i < j:
                        term1 = (mean_grdpg_r[i][j] - mean_grdpg_f[i][j]) ** 2
                        term2 = (var_grdpg_r[i][j] / m_r + var_grdpg_f[i][j] / m_f)
                        if term1 != 0 and term2 == 0:
                            p_value = 0
                            break
                        elif term2 == 0:
                            continue
                        nominator += term1
                        denominator += term2
            if p_value == -1:
                test_statistic = nominator / denominator
                test_statistic = np.nan_to_num(test_statistic)
                # Even though it evaluates the upper tail area, the chi-square test is regarded as a two-tailed test (non-directional)
                p_value = 1 - chi2.cdf(test_statistic, self.number_of_nodes * (self.number_of_nodes - 1) / 2)

        elif 2 <= sample_size < self.number_of_nodes:
            # proposed normality based test (Asymp-Normal)
            # https://github.com/gdebarghya/Network-TwoSampleTesting/blob/master/codes/NormalityTest.m

            m1 = floor(0.5 * min(m_r, m_f))
            term1_1 = np.sum(graph_samples_r[:m1, :, :] - graph_samples_f[:m1, :, :], axis=0)
            term1_2 = np.sum(graph_samples_r[m1:, :, :] - graph_samples_f[m1:, :, :], axis=0)
            term2_1 = np.sum(graph_samples_r[:m1, :, :] + graph_samples_f[:m1, :, :], axis=0)
            term2_2 = np.sum(graph_samples_r[m1:, :, :] + graph_samples_f[m1:, :, :], axis=0)
            nominator = 0
            denominator = 0
            for i in range(self.number_of_nodes):
                for j in range(self.number_of_nodes):
                    if i < j:
                        nominator += term1_1[i][j] * term1_2[i][j]
                        denominator += term2_1[i][j] * term2_2[i][j]
            test_statistic = nominator / np.sqrt(denominator)
            p_value = 2 * norm.cdf(-abs(test_statistic))

        elif sample_size == 1:
            # it is a type of Tracy-Widom test statistic
            # rewrite based on https://github.com/gdebarghya/Network-TwoSampleTesting/blob/master/codes/TracyWidomTest.m

            # convert to unweighted graphs
            idx_r0, idx_r1 = graph_samples_r > self.threshold_r, graph_samples_r <= self.threshold_r
            graph_samples_r[idx_r0], graph_samples_r[idx_r1] = 0, 1
            idx_f0, idx_f1 = graph_samples_f > self.threshold_f, graph_samples_f <= self.threshold_f
            graph_samples_f[idx_f0], graph_samples_f[idx_f1] = 0, 1

            c = graph_samples_r - graph_samples_f
            idx = self._spectral_clustering(graph_samples_r, graph_samples_f)
            for i in range(self.r):
                for j in range(self.r):
                    if i == j:
                        temp = graph_samples_r[np.ix_(i == idx, j == idx)]
                        # takes into account the diagonal is zero
                        Pij = 0 if temp.shape[0] <= 1 else np.sum(temp) / (temp.shape[0] * temp.shape[0] - 1)
                        temp = graph_samples_f[np.ix_(i == idx, j == idx)]
                        Qij = 0 if temp.shape[0] <= 1 else np.sum(temp) / (temp.shape[0] * temp.shape[0] - 1)
                        # continue
                    else:
                        temp = graph_samples_r[np.ix_(i == idx, j == idx)]
                        Pij = np.mean(temp)
                        temp = graph_samples_f[np.ix_(i == idx, j == idx)]
                        Qij = np.mean(temp)
                    denominator = np.sqrt((self.number_of_nodes - 1) * Pij * (1 - Pij) + Qij * (1 - Qij))
                    if denominator == 0:
                        denominator = 1e-5
                    c[np.ix_(i == idx, j == idx)] = c[np.ix_(i == idx, j == idx)] / denominator
            c = np.nan_to_num(c)
            u, s, et = np.linalg.svd(c)
            # using the largest singular value
            test_statistic = self.number_of_nodes ** (2 / 3) * (s[0] - 2)
            tw1_dist = TracyWidom(beta=1)
            p_value = min(1, 2 * (1 - tw1_dist.cdf(test_statistic)))
        return p_value

    def _clean(self):
        del self._samples_by_partition
        del self._partition_count
        gc.collect()
        self.size_in_memory = 0

    def _spectral_clustering(self, a, b):
        """
        spectral clustering to find common block structure
        a, b are two graphs, and r is a scalar (specifying number of communities)
        """
        c = (a + b) / 2
        d = np.sum(c, axis=1)
        d[d == 0] = 1
        d = 1 / np.sqrt(d)
        c = c * np.dot(d, d.T)
        u, s, et = np.linalg.svd(c)
        # using r largest dominant singular vectors
        e = np.array(et[:self.r, :], ndmin=2).T
        norm_e = np.array(np.sqrt(np.sum(e ** 2, axis=1)), ndmin=2).T
        norm_e[norm_e == 0] = 1
        e = e / norm_e
        km = KMeans(n_clusters=self.r, random_state=0).fit(e)
        return km.labels_


class GraphTestEngine_Attack(GuessEngine):
    """
    attack version of GraphTestEngine, pls ref to the DpaEngine, where the LSB can be used
    """

    def __init__(self, name, selection_function, guess_range, time_delay, dim, sampling_interval=1, r=3, solution=-1):
        """
        :param name:
        :param param selection_function: takes a value and a guess_guess as input, returns 0 or 1.
        :param guess_range: what are the values for the guess guess
        :param time_delay: delayed time interval used in phase space reconstruction
        :param dim: the dimension of embedding delayed time series (vectors)
        :param sampling_interval: used to sample the delayed time series, default is 1
        :param r: the number of communities (or rank) r only for approximation of P and Q used in the Tracy-Widom test
                  noted that the power of the test is not sensitive to the choice of r
        :param solution: if known, indicate the correct guess guess.
        """
        self.time_delay = time_delay
        self.dim = dim
        self.sampling_interval = sampling_interval
        self.r = r
        self.solution = solution
        GuessEngine.__init__(self, name, selection_function, guess_range, solution)
        self.output_parser_mode = "max"
        self.logger.debug(
            'Creating GraphTestEngine_Attack "%s" with %d guesses.', name, len(guess_range)
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
            p_value = -1

            set_0, set_1 = self._samples_by_selection[guess][0], self._samples_by_selection[guess][1]
            m_0, m_1 = self._count_x[guess, 0], self._count_x[guess, 1]

            # convert 1-d time series into 2-d graphs by phase space reconstruction
            # init_graph_0 = Phase_Space_Reconstruction_Graph(set_0, self.time_delay, self.dim, self.sampling_interval)
            # init_graph_0.generate()
            # init_graph_1 = Phase_Space_Reconstruction_Graph(set_1, self.time_delay, self.dim, self.sampling_interval)
            # init_graph_1.generate()

            init_graph_0 = Simple_PSRG(set_0, self.time_delay, self.dim, self.sampling_interval)
            init_graph_0.generate()
            init_graph_1 = Simple_PSRG(set_1, self.time_delay, self.dim, self.sampling_interval)
            init_graph_1.generate()

            # size of graph
            self.number_of_nodes = init_graph_0.number_of_nodes

            sample_size = (m_0 + m_1) // 2
            if sample_size > self.number_of_nodes:
                graph_samples_r = init_graph_0.adj_matrix
                graph_samples_f = init_graph_1.adj_matrix
                # graph_samples_f = np.ones(graph_samples_r.shape)
                # it is a type of chi-square test statistic
                mean_grdpg_r, mean_grdpg_f = np.mean(graph_samples_r, axis=0), np.mean(graph_samples_f, axis=0)
                var_grdpg_r, var_grdpg_f = np.var(graph_samples_r, axis=0), np.var(graph_samples_f, axis=0)

                nominator = 0
                denominator = 0
                for i in range(self.number_of_nodes):
                    for j in range(self.number_of_nodes):
                        if i < j:
                            term1 = (mean_grdpg_r[i][j] - mean_grdpg_f[i][j]) ** 2
                            term2 = (var_grdpg_r[i][j] / m_0 + var_grdpg_f[i][j] / m_1)
                            if term1 != 0 and term2 == 0:
                                p_value = 0
                                break
                            elif term2 == 0:
                                continue
                            nominator += term1
                            denominator += term2
                if p_value == -1:
                    test_statistic = nominator / denominator if denominator else 1
                    test_statistic = np.nan_to_num(test_statistic)
                    # Even though it evaluates the upper tail area, the chi-square test is regarded as a two-tailed test (non-directional)
                    p_value = 1 - chi2.cdf(test_statistic, self.number_of_nodes * (self.number_of_nodes - 1) / 2)

            elif 2 <= sample_size < self.number_of_nodes:
                # proposed normality based test (Asymp-Normal)
                # https://github.com/gdebarghya/Network-TwoSampleTesting/blob/master/codes/NormalityTest.m
                graph_samples_r = init_graph_0.adj_matrix
                graph_samples_f = init_graph_1.adj_matrix
                m = floor(0.5 * min(m_0, m_1))
                term1_1 = np.sum(graph_samples_r[:0, :, :] - graph_samples_f[:0, :, :], axis=0)
                term1_2 = np.sum(graph_samples_r[0:, :, :] - graph_samples_f[0:, :, :], axis=0)
                term2_1 = np.sum(graph_samples_r[:0, :, :] + graph_samples_f[:0, :, :], axis=0)
                term2_2 = np.sum(graph_samples_r[0:, :, :] + graph_samples_f[0:, :, :], axis=0)
                nominator = 0
                denominator = 0
                for i in range(self.number_of_nodes):
                    for j in range(self.number_of_nodes):
                        if i < j:
                            nominator += term1_1[i][j] * term1_2[i][j]
                            denominator += term2_1[i][j] * term2_2[i][j]
                test_statistic = nominator / np.sqrt(denominator)
                p_value = 2 * norm.cdf(-abs(test_statistic))

            elif sample_size == 1:
                # it is a type of Tracy-Widom test statistic
                # rewrite based on https://github.com/gdebarghya/Network-TwoSampleTesting/blob/master/codes/TracyWidomTest.m
                graph_samples_r = init_graph_0.adj_matrix[0]
                graph_samples_f = init_graph_1.adj_matrix[0]
                c = graph_samples_r - graph_samples_f
                idx = self._spectral_clustering(graph_samples_r, graph_samples_f)
                for i in range(self.r):
                    for j in range(self.r):
                        if i == j:
                            temp = graph_samples_r[np.ix_(i == idx, j == idx)]
                            # takes into account the diagonal is zero
                            Pij = 0 if temp.shape[0] <= 1 else np.sum(temp) / (temp.shape[0] * temp.shape[0] - 1)
                            temp = graph_samples_f[np.ix_(i == idx, j == idx)]
                            Qij = 0 if temp.shape[0] <= 1 else np.sum(temp) / (temp.shape[0] * temp.shape[0] - 1)
                            # continue
                        else:
                            temp = graph_samples_r[np.ix_(i == idx, j == idx)]
                            Pij = np.mean(temp)
                            temp = graph_samples_f[np.ix_(i == idx, j == idx)]
                            Qij = np.mean(temp)
                        denominator = np.sqrt((self.number_of_nodes - 1) * Pij * (1 - Pij) + Qij * (1 - Qij))
                        if denominator == 0:
                            denominator = 1e-5
                        c[np.ix_(i == idx, j == idx)] = c[np.ix_(i == idx, j == idx)] / denominator
                c = np.nan_to_num(c)
                u, s, et = np.linalg.svd(c)
                # using the largest singular value
                test_statistic = self.number_of_nodes ** (2 / 3) * (s[0] - 2)
                tw1_dist = TracyWidom(beta=1)
                p_value = min(1, 2 * (1 - tw1_dist.cdf(test_statistic)))
            self._test_results[0, guess] = p_value
        return self._test_results

    def _clean(self):
        del self._samples_by_selection
        del self._count_x
        gc.collect()

    def _spectral_clustering(self, a, b):
        """
        spectral clustering to find common block structure
        a, b are two graphs, and r is a scalar (specifying number of communities)
        """
        c = (a + b) / 2
        d = np.sum(c, axis=1)
        d[d == 0] = 1
        d = 1 / np.sqrt(d)
        c = c * np.dot(d, d.T)
        u, s, et = np.linalg.svd(c)
        # using r largest dominant singular vectors
        e = np.array(et[:self.r, :], ndmin=2).T
        norm_e = np.array(np.sqrt(np.sum(e ** 2, axis=1)), ndmin=2).T
        norm_e[norm_e == 0] = 1
        e = e / norm_e
        km = KMeans(n_clusters=self.r, random_state=0).fit(e)
        return km.labels_


class TraceBasedGraphTestEngine(PartitionerEngine):
    """
        trace based version of GraphTestEngine
        we convert the whole traces to a graph, one trace represents one node and correlation will be calculated for connectivity
    """

    def __init__(self, name, partition_function, r=3):
        """
        :param name:
        :param partition_function: partition_function that will take trace values as an input and returns 0 or 1
        :param r: the number of communities (or rank) r only for approximation of P and Q used in the Tracy-Widom test
                  noted that the power of the test is not sensitive to the choice of r
        """
        self.r = r

        self.threshold_rs = list()
        self.threshold_fs = list()

        PartitionerEngine.__init__(self, name, partition_function, range(2), None)
        self.logger.debug('Creating GraphTestEngine  "%s". ' % name)

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

        # record each threshold
        self.threshold_rs.append(self._calc_best_conversion_threshold(adj_matrix_r))
        self.threshold_fs.append(self._calc_best_conversion_threshold(adj_matrix_f))

    def _finalize(self):
        p_value = -1.0
        m_r, m_f = len(self._graph_set_r), len(self._graph_set_f)
        sample_size = (m_r + m_f) // 2

        graph_samples_r = np.array(self._graph_set_r) if sample_size != 1 else self._graph_set_r[0]
        graph_samples_f = np.array(self._graph_set_f) if sample_size != 1 else self._graph_set_f[0]

        self.threshold_r = np.mean(self.threshold_rs)
        self.threshold_f = np.mean(self.threshold_fs)

        # convert to unweighted graphs
        idx_r0, idx_r1 = np.array(self._graph_set_r) > self.threshold_r, graph_samples_r <= self.threshold_r
        graph_samples_r[idx_r0], graph_samples_r[idx_r1] = 0, 1
        idx_f0, idx_f1 = graph_samples_f > self.threshold_f, graph_samples_f <= self.threshold_f
        graph_samples_f[idx_f0], graph_samples_f[idx_f1] = 0, 1

        if sample_size > self.number_of_nodes:
            # graph_samples_f = np.ones(graph_samples_r.shape)
            # it is a type of chi-square test statistic
            mean_grdpg_r, mean_grdpg_f = np.mean(graph_samples_r, axis=0), np.mean(graph_samples_f, axis=0)
            var_grdpg_r, var_grdpg_f = np.var(graph_samples_r, axis=0), np.var(graph_samples_f, axis=0)

            nominator = 0
            denominator = 0
            for i in range(self.number_of_nodes):
                for j in range(self.number_of_nodes):
                    if i < j:
                        term1 = (mean_grdpg_r[i][j] - mean_grdpg_f[i][j]) ** 2
                        term2 = (var_grdpg_r[i][j] / m_r + var_grdpg_f[i][j] / m_f)
                        if term1 != 0 and term2 == 0:
                            p_value = 0
                            break
                        elif term2 == 0:
                            continue
                        nominator += term1
                        denominator += term2
            if p_value == -1:
                test_statistic = nominator / denominator
                test_statistic = np.nan_to_num(test_statistic)
                # Even though it evaluates the upper tail area, the chi-square test is regarded as a two-tailed test (non-directional)
                p_value = 1 - chi2.cdf(test_statistic, self.number_of_nodes * (self.number_of_nodes - 1) / 2)

        elif 2 <= sample_size < self.number_of_nodes:
            # proposed normality based test (Asymp-Normal)
            # https://github.com/gdebarghya/Network-TwoSampleTesting/blob/master/codes/NormalityTest.m
            m1 = floor(0.5 * min(m_r, m_f))
            term1_1 = np.sum(graph_samples_r[:m1, :, :] - graph_samples_f[:m1, :, :], axis=0)
            term1_2 = np.sum(graph_samples_r[m1:, :, :] - graph_samples_f[m1:, :, :], axis=0)
            term2_1 = np.sum(graph_samples_r[:m1, :, :] + graph_samples_f[:m1, :, :], axis=0)
            term2_2 = np.sum(graph_samples_r[m1:, :, :] + graph_samples_f[m1:, :, :], axis=0)
            nominator = 0
            denominator = 0
            for i in range(self.number_of_nodes):
                for j in range(self.number_of_nodes):
                    if i < j:
                        nominator += term1_1[i][j] * term1_2[i][j]
                        denominator += term2_1[i][j] * term2_2[i][j]
            test_statistic = nominator / np.sqrt(denominator)
            p_value = 2 * norm.cdf(-abs(test_statistic))

        elif sample_size == 1:
            # it is a type of Tracy-Widom test statistic
            # rewrite based on https://github.com/gdebarghya/Network-TwoSampleTesting/blob/master/codes/TracyWidomTest.m
            c = graph_samples_r - graph_samples_f
            idx = self._spectral_clustering(graph_samples_r, graph_samples_f)
            for i in range(self.r):
                for j in range(self.r):
                    if i == j:
                        temp = graph_samples_r[np.ix_(i == idx, j == idx)]
                        # takes into account the diagonal is zero
                        Pij = 0 if temp.shape[0] <= 1 else np.sum(temp) / (temp.shape[0] * temp.shape[0] - 1)
                        temp = graph_samples_f[np.ix_(i == idx, j == idx)]
                        Qij = 0 if temp.shape[0] <= 1 else np.sum(temp) / (temp.shape[0] * temp.shape[0] - 1)
                        # continue
                    else:
                        temp = graph_samples_r[np.ix_(i == idx, j == idx)]
                        Pij = np.mean(temp)
                        temp = graph_samples_f[np.ix_(i == idx, j == idx)]
                        Qij = np.mean(temp)
                    denominator = np.sqrt((self.number_of_nodes - 1) * Pij * (1 - Pij) + Qij * (1 - Qij))
                    if denominator == 0:
                        denominator = 1e-5
                    c[np.ix_(i == idx, j == idx)] = c[np.ix_(i == idx, j == idx)] / denominator
            c = np.nan_to_num(c)
            u, s, et = np.linalg.svd(c)
            # using the largest singular value
            test_statistic = self.number_of_nodes ** (2 / 3) * (s[0] - 2)
            tw1_dist = TracyWidom(beta=1)
            p_value = min(1, 2 * (1 - tw1_dist.cdf(test_statistic)))
        # self._clean()
        return p_value

    def _clean(self):
        del self._samples_by_partition
        del self._partition_count
        gc.collect()

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

    def _calc_best_conversion_threshold(self, g):
        """
        inputs: one adjacent matrix of weighted graph
        using the graph density to choose the optimal threshold
        for undirected graph, density = 2 * m / n * (n - 1); for directed graph, density = m / n * (n - 1)
        m-number of edges; n-number of nodes
        """
        potential_thresholds = np.arange(np.min(g[g > 0]), np.max(g[g > 0]), 0.1)
        best_threshold = g[0][0]
        prev_density = 0
        best_diff = float('-inf')
        for i, threshold in enumerate(potential_thresholds):
            tmp = np.copy(g)
            idx_0 = tmp > threshold
            idx_1 = tmp <= threshold
            tmp[idx_0], tmp[idx_1] = 0, 1
            tmp[np.diag_indices_from(tmp)] = 0
            num_edges = np.count_nonzero(np.triu(tmp, 1) == 1)
            curr_density = (2 * num_edges) / (self.number_of_nodes * (self.number_of_nodes - 1))
            cur_diff = 0 if i == 0 else (curr_density - prev_density)
            if cur_diff > best_diff:
                best_diff = cur_diff
                best_threshold = threshold
            prev_density = curr_density
        return best_threshold

    def _spectral_clustering(self, a, b):
        """
        spectral clustering to find common block structure
        a, b are two graphs, and r is a scalar (specifying number of communities)
        """
        c = (a + b) / 2
        d = np.sum(c, axis=1)
        d[d == 0] = 1
        d = 1 / np.sqrt(d)
        c = c * np.dot(d, d.T)
        u, s, et = np.linalg.svd(c)
        # using r largest dominant singular vectors
        e = np.array(et[:self.r, :], ndmin=2).T
        norm_e = np.array(np.sqrt(np.sum(e ** 2, axis=1)), ndmin=2).T
        norm_e[norm_e == 0] = 1
        e = e / norm_e
        km = KMeans(n_clusters=self.r, random_state=0).fit(e)
        return km.labels_
