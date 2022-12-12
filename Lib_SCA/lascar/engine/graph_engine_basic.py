import numpy as np
from scipy.stats import norm, bernoulli


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
        max_dim = self.number_of_time_points // self.time_delay + 1
        xxx = self._false_nearest_neighbor(self.time_series[0, :], self.time_delay, max_dim)

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

    def _c_c_method(self):
        """
        using c-c method to determine the optimal time delay for phase space reconstruction
        ref: Kim, H_S, R. Eykholt, and J. D. Salas. "Nonlinear dynamics, delay times, and embedding windows."
        Physica D: Nonlinear Phenomena 127.1-2 (1999): 48-60.
        """
        pass

    def _false_nearest_neighbor(self, one_series, time_delay, max_dim, rtol=9, atol=2):
        """
        using false nearest neighbors (fnn) to determine the optimal embedding dimension of delayed subsequences
        ref: M. B. Kennel, R. Brown, and H. D. I. Abarbanel, Determining embedding dimension for phase-space reconstruction
        using a geometrical construction, Phys. Rev. A 45, 3403 (1992).

        :param rtol: threshold for the 1st criterion
        :param atol: threshold for the 2nd criterion
        """
        n = one_series.shape[0]
        Ra = np.std(one_series)
        fnn = np.zeros(max_dim)
        for dim in range(1, max_dim + 1):
            space_points = n - (dim - 1) * time_delay // self.sampling_interval
            y = self._psr_single_series(one_series, dim, time_delay)
            count = 0
            for sp_i in range(space_points):
                y0 = np.dot(np.ones((space_points, 1)), np.array(y[sp_i, :], ndmin=2))
                dist = np.sqrt(np.sum((y - y0) ** 2, axis=1))
                sorted_dist, sorted_idx = np.sort(dist), np.argsort(dist)
                if sp_i + dim * time_delay < n and sorted_idx[1] + dim * time_delay < n:
                    count += 1
                    D = abs(one_series[sp_i + dim * time_delay] - one_series[sorted_idx[1] + dim * time_delay])
                    R = np.sqrt(D ** 2 + sorted_dist[1] ** 2)
                    if D / sorted_dist[1] > rtol or R / Ra > atol:
                        fnn[dim - 1] += 1
            fnn[dim - 1] = fnn[dim - 1] / count if count else 1
            if not fnn[dim - 1]:
                return dim
        return np.argmin(fnn) + 1

    def _psr_single_series(self, one_series, dim, time_delay):
        n = one_series.shape[0]
        space_points = n - (dim - 1) * time_delay // self.sampling_interval
        res = np.zeros((space_points, dim))
        for i in range(space_points):
            res[i, :] = one_series[i: i + (dim - 1) * time_delay + 1:time_delay]
        return res


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