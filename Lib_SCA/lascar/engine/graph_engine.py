import gc
from math import floor

import numpy as np
from scipy.stats import norm, bernoulli, chi2
from sklearn.cluster import KMeans
from TracyWidom import TracyWidom
import sklearn.feature_selection as fs
from tqdm import tqdm
import networkx as nx

from . import PartitionerEngine


class Phase_Space_Reconstruction_Graph:
    """
    constructing 2-d graphs from 1-d time series with each vector point of the reconstructed phase space represented
    by a single node and edge determined by the phase space distance
    in the phase space reconstruction method, a phase space trajectory can be reconstructed from a time series by time
    delay embedding, and each time delayed series (vectors) generated from the original time series can represent one
    possible state (or phase) of the system

    original paper:
    Gao, Zhongke, and Ningde Jin. "Complex network from time series based on phase space reconstruction."
    Chaos: An Interdisciplinary Journal of Nonlinear Science 19.3 (2009): 033137.
    """

    def __init__(self, time_series, time_delay, dim, sampling_interval=1, to_unweighted=True):
        """
        Phase_Space_Reconstruction_Graph
        :param time_series: original 1-d time series collections (a 2-d ndarray which is # time series * # time points)
        :param time_delay: delayed time interval used in phase space reconstruction
        :param dim: the dimension of embedding delayed time series (vectors)
        :param sampling_interval: used to sample the delayed time series, default is 1
        :params to_unweighted: whether or not convert to unweighted graphs
        :param number_of_nodes: number of nodes in the generated graph, which is equal to the number of embedding
                                vectors in the reconstructed phase space
        :param adj_matrix: resulting adjacent matrix of the generated graph
        """
        self.time_series = time_series
        self.number_of_time_series = self.time_series.shape[0]
        self.number_of_time_points = self.time_series.shape[1]

        self.time_delay = time_delay
        self.dim = dim
        self.sampling_interval = sampling_interval
        self.to_unweighted = to_unweighted

        self.number_of_nodes = self.number_of_time_points - (self.dim - 1) * self.time_delay // self.sampling_interval
        self.adj_matrix = np.zeros((self.number_of_time_series, self.number_of_nodes, self.number_of_nodes),
                                   dtype=np.float64)

    def generate(self):
        """
        the time delayed embedding process
        """
        for i in range(self.number_of_time_series):
            tmp = np.zeros((self.number_of_nodes, self.dim))
            for j in range(self.number_of_nodes):
                tmp[j, :] = self.time_series[i, j:j + (self.dim - 1) * self.time_delay + 1:self.time_delay]
            weighted_adj = self._calc_euclidean_distances(tmp, tmp)
            self.adj_matrix[i, :, :] = self._convert_to_unweighted_graphs(
                weighted_adj) if self.to_unweighted else weighted_adj

    @staticmethod
    def _calc_euclidean_distances(a, b):
        abt = np.dot(a, b.T)
        a2 = np.array(np.sum(a ** 2, axis=1), ndmin=2).T.repeat(abt.shape[1], axis=1)
        b2 = np.array(np.sum(b ** 2, axis=1), ndmin=2).repeat(abt.shape[0], axis=0)
        tmp = a2 + b2 - 2 * abt
        tmp[tmp <= 1e-9] = 1e-9
        res = np.sqrt(tmp)
        res[np.diag_indices_from(res)] = 0
        return np.nan_to_num(res)

    def _convert_to_unweighted_graphs(self, g):
        """
        inputs: one adjacent matrix of weighted graph
        using the graph density to choose the optimal threshold
        for undirected graph, density = 2 * m / n * (n - 1); for directed graph, density = m / n * (n - 1)
        m-number of edges; n-number of nodes
        """
        potential_threshold = np.arange(np.min(g[g > 0]), np.max(g[g > 0]), 0.1)
        best_g = 0
        prev_density = 0
        best_diff = float('-inf')
        for i, threshold in enumerate(potential_threshold):
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
                best_g = tmp
            prev_density = curr_density
        return best_g


class Generalised_RDPG:
    """
    random dot product graph (RDPG) is a kind of random graph where each node is associated with a latent space (vector)
    and the probability between each pair of nodes is simply defined by the dot product result of their associated latent
    vectors

    latent vectors can be found by spectral decomposition of given adjacent matrix with selecting d largest eigenvalues
    and it should be under a positive-definite assumption, which is known as Adjacency Spectral Embedding (ASE).
    however such a model must result in a non-negative-definite edge probability matrix and cannot explain significant
    negative eigenvalues in the original adjacent matrix

    Generalised RDPG (GRDPG) is designed to tackle with this issue is the paper:
    Rubin-Delanchy, Patrick, et al. "A statistical interpretation of spectral embedding: the generalised random dot product graph."
    arXiv preprint arXiv:1709.05506 (2017).

    the finalized edge occurrence follows a bernoulli distribution which is the same as inhomogeneous ER (IER) graphs
    """

    def __init__(self, graph_samples, mode='single'):
        """
        Generalised_RDPG
        : param graph_samples: a sequence of graphs (a 3-d ndarray which is # graphs * 2-d adjacent matrix)
        : param mode: 'multi'-convert each graph into a GRDPG; 'single'-convert all graph samples into a GRDPG
        """
        self.graph_samples = graph_samples
        self.number_of_graphs = self.graph_samples.shape[0]
        self.mode = mode

    def generate(self):
        grdpg = 0
        if self.mode == 'multi':
            for i in range(self.number_of_graphs):
                grdpg = np.zeros(self.graph_samples.shape)
                adj_matrix = self.graph_samples[i, :, :]
                xt, identity_pq = self._ase_of_grdpg(adj_matrix)
                xt_norm = np.linalg.norm(xt)  # normalization
                xt = xt / xt_norm
                x = xt.T
                p = np.real(np.dot(np.dot(xt, identity_pq), x))
                grdpg[i, :, :] = bernoulli.rvs(p, size=(self.graph_samples.shape[1], self.graph_samples.shape[2]))
        elif self.mode == 'single':
            adj_matrix = np.sum(self.graph_samples, axis=0)
            xt, identity_pq = self._ase_of_grdpg(adj_matrix)
            xt_norm = np.linalg.norm(xt)  # normalization
            xt = xt / xt_norm
            x = xt.T
            p = np.real(np.dot(np.dot(xt, identity_pq), x))
            # grdpg = bernoulli.rvs(p, size=(self.graph_samples.shape[1], self.graph_samples.shape[2]))
            grdpg = p
        return grdpg

    @staticmethod
    def _ase_of_grdpg(adj_matrix):
        """
        using profile log-likelihood to determine the best d in spectral decomposition
        Zhu, Mu, and Ali Ghodsi. "Automatic dimensionality selection from the scree plot via the use of profile likelihood."
        Computational Statistics & Data Analysis 51.2 (2006): 918-930.
        """
        eig_value, eig_vector = np.linalg.eig(adj_matrix)
        idx = np.argsort(eig_value)[::-1]
        eig_value = eig_value[idx]
        eig_vector = eig_vector[:, idx]
        r = eig_value.shape[0]
        best_d = 0
        score = float('-inf')
        for d in range(1, r):
            g1, g2 = eig_value[:d], eig_value[d:]
            m1, m2 = np.mean(g1), np.mean(g2)
            s1, s2 = np.var(g1), np.var(g2)
            std = np.sqrt(((d - 1) * s1 + (r - d - 1) * s2) / (r - 2))
            profile_ll = np.sum(np.log(norm.pdf(eig_value[:d], loc=m1, scale=std))) + np.sum(
                np.log(norm.pdf(eig_value[d:], loc=m2, scale=std)))
            if profile_ll > score:
                score = profile_ll
                best_d = d
        best_eig_value = eig_value[:best_d]
        gamma = np.diag(best_eig_value)
        best_eig_vector = eig_vector[:, :best_d]
        xt = np.dot(best_eig_vector, np.sqrt(gamma))  # latent vector
        p = np.sum(best_eig_value > 0)
        q = best_d - p
        identity_pq = np.diag([1] * p + [-1] * q)
        return xt, identity_pq


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
        p_value = -1
        random_set, fixed_set = self._samples_by_partition[0], self._samples_by_partition[1]
        m_r, m_f = self._partition_count[0], self._partition_count[1]

        # convert 1-d time series into 2-d graphs by phase space reconstruction
        init_graph_r = Phase_Space_Reconstruction_Graph(random_set, self.time_delay, self.dim, self.sampling_interval)
        init_graph_r.generate()
        init_graph_f = Phase_Space_Reconstruction_Graph(fixed_set, self.time_delay, self.dim, self.sampling_interval)
        init_graph_f.generate()

        # size of graph
        self.number_of_nodes = init_graph_r.number_of_nodes

        # convert determinate graphs into generalised RDPG which is also a type of inhomogeneous ER (IER) graphs
        # grdpg_r = Generalised_RDPG(init_graph_r.adj_matrix, self.mode).generate()
        # grdpg_f = Generalised_RDPG(init_graph_f.adj_matrix, self.mode).generate()

        sample_size = (m_r + m_f) // 2
        if sample_size > self.number_of_nodes:
            graph_samples_r = init_graph_r.adj_matrix
            graph_samples_f = init_graph_f.adj_matrix
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
            graph_samples_r = init_graph_r.adj_matrix
            graph_samples_f = init_graph_f.adj_matrix
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
            graph_samples_r = init_graph_r.adj_matrix[0]
            graph_samples_f = init_graph_f.adj_matrix[0]
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
        print(p_value)

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


class GraphMIEngine(PartitionerEngine):
    """
        GraphMIEngine is a graph-based mi engine, used to calculate random graphs with known distribution (pdf of pmf)
        and form a cmi-like statistical inference to obtain p-value as well

        random graphs is a type of , obtained from a generalised RDPG construction with 'single' mode
    """

    def __init__(self, name, partition_function, time_delay, dim, sampling_interval=1, mi_mode='simple', num_shuffles=50):
        """
        :param name:
        :param partition_function: partition_function that will take trace values as an input and returns 0 or 1
        :param time_delay: delayed time interval used in phase space reconstruction
        :param dim: the dimension of embedding delayed time series (vectors)
        :param sampling_interval: used to sample the delayed time series, default is 1
        :param mi_mode: if 'simple', using discrete mi based on unweighted graphs;
                          if 'sklearn', using the existing method provided by the sklearn lib;
                          if 'estimate', using the method proposed in Escolano, Francisco, et al. "The mutual information between graphs." Pattern Recognition Letters 87 (2017): 12-19.
        :param num_shuffles: random shuffle times used to obtain the reasonable statistical test value
        """
        self.time_delay = time_delay
        self.dim = dim
        self.sampling_interval = sampling_interval
        if mi_mode == 'simple':
            self.mi_method = self._calc_graph_mutual_information_simple
        elif mi_mode == 'sklearn':
            self.mi_method = self._calc_graph_mutual_information_sklearn
        elif mi_mode == 'estimate':
            self.mi_method = self._calc_graph_mutual_information_estimate

        self.num_shuffles = num_shuffles
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

        graph_samples_r = init_graph_r.adj_matrix
        graph_samples_f = init_graph_f.adj_matrix

        # real mi
        real_mi = self.mi_method(graph_samples_r, graph_samples_f)

        # reference mi (same distribution) used in statistical testing (assuming a gaussian distribution)
        # both samples randomly drawn from a inhomogeneous ER random graph with each edge is a independent r.v.
        # follows a Bernoulli distribution (p=0.5)
        reference_mi = np.zeros(self.num_shuffles)
        # p = 0.5 * np.ones((m_r, self.number_of_nodes, self.number_of_nodes))
        for i in tqdm(range(self.num_shuffles)):
            np.random.shuffle(random_set)
            shuffled_graph_r = Phase_Space_Reconstruction_Graph(random_set, self.time_delay, self.dim, self.sampling_interval)
            shuffled_graph_r.generate()
            shuffled_graph_samples_r = shuffled_graph_r.adj_matrix
            # ier_graph_a = bernoulli.rvs(p, size=(m_r, self.number_of_nodes, self.number_of_nodes))
            # ier_graph_b = bernoulli.rvs(p, size=(m_f, self.number_of_nodes, self.number_of_nodes))
            reference_mi[i] = self.mi_method(graph_samples_r, shuffled_graph_samples_r)

        # p value calculation
        m = np.mean(reference_mi)
        v = np.std(reference_mi)
        p_value = 2 * norm.cdf(real_mi, loc=m, scale=v) if real_mi < m else 2 * (1 - norm.cdf(real_mi, loc=m, scale=v))
        print(real_mi, p_value)

    @staticmethod
    def _calc_simple_mi(x, y):
        pdf_matrix = np.zeros((3, 3))
        l = min(x.shape[0], y.shape[0])
        for i in range(l):
            state_a, state_b = int(x[i]), int(y[i])
            pdf_matrix[state_a, state_b] += 1

        pdf_matrix[0, 2] = pdf_matrix[0, 0] + pdf_matrix[0, 1]
        pdf_matrix[1, 2] = pdf_matrix[1, 0] + pdf_matrix[1, 1]
        pdf_matrix[2, 0] = pdf_matrix[0, 0] + pdf_matrix[1, 0]
        pdf_matrix[2, 1] = pdf_matrix[0, 1] + pdf_matrix[1, 1]
        pdf_matrix[2, 2] = l

        pdf_matrix /= l
        p00, p01, p10, p11 = pdf_matrix[0, 0], pdf_matrix[0, 1], pdf_matrix[1, 0], pdf_matrix[1, 1]
        p0_, p1_, p_0, p_1 = pdf_matrix[0, 2], pdf_matrix[1, 2], pdf_matrix[2, 0], pdf_matrix[2, 1]
        t00 = p00 * np.log(p00 / (p0_ * p_0)) if p00 > 0 else 0
        t01 = p01 * np.log(p01 / (p0_ * p_1)) if p01 > 0 else 0
        t10 = p10 * np.log(p10 / (p1_ * p_0)) if p10 > 0 else 0
        t11 = p11 * np.log(p11 / (p1_ * p_1)) if p11 > 0 else 0

        mi = t00 + t01 + t10 + t11
        return mi

    def _calc_graph_mutual_information_simple(self, graph_samples_a, graph_samples_b):
        mi = 0.0
        for i in range(self.number_of_nodes):
            for j in range(self.number_of_nodes):
                if i < j:
                    a = np.array(graph_samples_a[:, i, j], ndmin=2).T
                    b = graph_samples_b[:, i, j]
                    mi += self._calc_simple_mi(a, b)
        return mi

    def _calc_graph_mutual_information_sklearn(self, graph_samples_a, graph_samples_b):
        mi = 0.0
        for i in range(self.number_of_nodes):
            for j in range(self.number_of_nodes):
                if i < j:
                    a = np.array(graph_samples_a[:, i, j], ndmin=2).T
                    b = graph_samples_b[:, i, j]
                    mi += fs.mutual_info_classif(a, b)[0]
        return mi

    def _calc_graph_mutual_information_estimate(self):
        # TODO: a bit complicated
        pass

    def _clean(self):
        del self._samples_by_partition
        del self._partition_count
        gc.collect()
        self.size_in_memory = 0
