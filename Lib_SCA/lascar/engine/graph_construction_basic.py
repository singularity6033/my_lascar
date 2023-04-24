import collections
from bisect import bisect_left

import numpy as np
from tqdm import tqdm
from scipy.stats import norm, bernoulli, pearsonr, moment
from PyAstronomy import pyaC
from math import inf


class BaseGraph:
    def __init__(self, type='corr'):
        if type == 'corr':
            self.measurement_func = self.calc_correlation_matrix
        elif type == 'l2':
            self.measurement_func = self.calc_euclidean_distances
        self.type = type

    @staticmethod
    def calc_graph_laplacian_matrix(g):
        # g is an adjacent matrix
        d = np.diag(np.sum(g, axis=1)) - np.diag(g)
        l = d - g
        return l

    @staticmethod
    def calc_correlation_matrix(a, b):
        if a.shape[0] != b.shape[0]:
            print('a, b should have the same size along the first axis')
            return
        tmp_a, tmp_b = a.sum(0), b.sum(0)
        tmp_a2, tmp_b2 = (a ** 2).sum(0), (b ** 2).sum(0)
        tmp_ab = np.dot(a.T, b) / a.shape[0]
        v_a = tmp_a2 / a.shape[0] - (tmp_a / a.shape[0]) ** 2
        v_b = tmp_b2 / b.shape[0] - (tmp_b / b.shape[0]) ** 2
        numerator = tmp_ab - np.outer(tmp_a / a.shape[0], tmp_b / b.shape[0])
        denominator = np.sqrt(np.outer(v_a, v_b))
        mask_a, mask_b = v_a == 0.0, v_b == 0.0
        numerator[mask_a, mask_b], denominator[mask_a, mask_b] = 0.0, 1.0
        adj_matrix = np.nan_to_num(numerator / denominator)
        return np.abs(adj_matrix)

    @staticmethod
    def calc_euclidean_distances(a, b):
        abt = np.dot(a, b.T)
        a2 = np.array(np.sum(a ** 2, axis=1), ndmin=2).T.repeat(abt.shape[1], axis=1)
        b2 = np.array(np.sum(b ** 2, axis=1), ndmin=2).repeat(abt.shape[0], axis=0)
        tmp = a2 + b2 - 2 * abt
        tmp[tmp <= 1e-9] = 1e-9
        res = np.sqrt(tmp)
        res[np.diag_indices_from(res)] = 0
        return np.nan_to_num(res, nan=inf)

    def calc_best_conversion_threshold(self, g, num_of_nodes):
        """
            inputs: one adjacent matrix of weighted graph
            using the graph density to choose the optimal threshold
            for undirected graph, density = 2 * m / n * (n - 1); for directed graph, density = m / n * (n - 1)
            m-number of edges; n-number of nodes
        """
        g = abs(g)
        upper_tri_idx = np.triu_indices(num_of_nodes, 1)
        g = g[upper_tri_idx]
        min_g, max_g = np.min(g), np.max(g)
        step = (max_g - min_g) / 50
        potential_thresholds = np.arange(min_g, max_g, step)
        # potential_thresholds.append(max_g)
        best_threshold = g[0]
        prev_density = 0
        best_diff = float('-inf')
        for i, threshold in enumerate(potential_thresholds):
            tmp = np.copy(g)
            idx_0 = tmp > threshold
            idx_1 = tmp < threshold
            if self.type == 'corr':
                tmp[idx_0], tmp[idx_1] = 1, 0
            elif self.type == 'dist':
                tmp[idx_0], tmp[idx_1] = 0, 1
            tmp[tmp == threshold] = 0
            num_edges = np.count_nonzero(tmp == 1)
            curr_density = (2 * num_edges) / (num_of_nodes * (num_of_nodes - 1))
            cur_diff = 0 if i == 0 else abs((curr_density - prev_density))
            if cur_diff > best_diff:
                best_diff = cur_diff
                best_threshold = threshold
            prev_density = curr_density
        return best_threshold


class PhaseSpaceReconstructionGraph(BaseGraph):
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

    def __init__(self,
                 time_delay,
                 dim,
                 sampling_interval=1,
                 type='corr',
                 optimal=False,
                 to_unweighted=False,
                 ):
        """
            Phase_Space_Reconstruction_Graph
            :param time_delay: delayed time interval used in phase space reconstruction
            :param dim: the dimension of embedding delayed time series (vectors)
            :param sampling_interval: used to sample the delayed time series, default is 1
            :param type: different measurements to calculate connectivity among nodes (embedded vectors); 'l2' ->
                         2-norm, 'corr' -> pearson correlation coefficient
            :param optimal: False -> use pre-defined delayed time step and embedded dimension in PSRG; True -> use c-c method
                         and fnn method to determine the best value for delayed time step and embedded dimension automatically
            :param to_unweighted: save thresholds or not
        """
        self.time_delay = time_delay
        self.dim = dim
        self.sampling_interval = sampling_interval

        self.optimal = optimal
        self.to_unweighted = to_unweighted

        BaseGraph.__init__(self, type=type)

    def generate(self, one_time_series):
        """
        the time delayed embedding process
        """
        number_of_time_points = one_time_series.shape[0]
        if self.optimal:
            self.time_delay = int(self._c_c_method(one_time_series, number_of_time_points))
            self.dim = self._false_nearest_neighbor(one_time_series, self.time_delay)
            number_of_nodes = number_of_time_points - (self.dim - 1) * self.time_delay // self.sampling_interval
        else:
            if number_of_time_points < (self.dim - 1) * self.time_delay // self.sampling_interval:
                print('[INFO] dimension size or delayed time is set too large, pls check the configuration files')
                return
            number_of_nodes = number_of_time_points - (self.dim - 1) * self.time_delay // self.sampling_interval
        vector_list = self._psr_single_series(one_time_series, self.dim, self.time_delay)
        weighted_adj = self.measurement_func(vector_list, vector_list)

        w_to_unw_threshold = None
        if self.to_unweighted:
            w_to_unw_threshold = self.calc_best_conversion_threshold(weighted_adj, number_of_nodes)
        return weighted_adj, w_to_unw_threshold

    def _c_c_method(self, one_series, number_of_time_points):
        """
        using c-c method to determine the optimal time delay for phase space reconstruction
        ref: Kim, H_S, R. Eykholt, and J. D. Salas. "Nonlinear dynamics, delay times, and embedding windows."
        Physica D: Nonlinear Phenomena 127.1-2 (1999): 48-60.
        """
        n = one_series.shape[0]
        std = np.std(one_series)
        r = [0.5 * std, 1.0 * std, 1.5 * std, 2.0 * std]
        max_t = min(number_of_time_points // (5 - 1), 10)  # control the upper limit to 5
        t_series = np.arange(1, max_t)  # exclude max
        res = np.ones(t_series.shape[0])

        for t_i, t in enumerate(t_series):
            delta_s = np.zeros(4)
            for dim_i, dim in enumerate(range(2, 6)):
                space_points = n - (dim - 1) * t // self.sampling_interval
                y = self._psr_single_series(one_series, dim, t)
                sup_norm_res = self._calc_sup_norm(y)
                sn = np.tile(sup_norm_res, (len(r), 1))
                rj = np.repeat(np.array(r, ndmin=2).T, sup_norm_res.shape[0], axis=1)
                sum_sup_norm_res = np.sum(np.heaviside(rj - sn, 1.0), axis=1)
                s = sum_sup_norm_res / ((space_points - 1) * space_points)
                delta_s[dim_i] = np.max(s) - np.min(s)
            res[t_i] = np.sum(delta_s) / 4

        # find first local minima
        extrema = np.diff(np.sign(np.diff(res)))
        minima = np.argwhere(extrema < 0)
        return t_series[minima[0] + 1] if minima.shape[0] else t_series[0]

    def _false_nearest_neighbor(self, one_series, time_delay, rtol=15, atol=2):
        """
        using false nearest neighbors (fnn) to determine the optimal embedding dimension of delayed subsequences
        ref: M. B. Kennel, R. Brown, and H. D. I. Abarbanel, Determining embedding dimension for phase-space reconstruction
        using a geometrical construction, Phys. Rev. A 45, 3403 (1992).

        :param rtol: threshold for the 1st criterion
        :param atol: threshold for the 2nd criterion
        """
        n = one_series.shape[0]
        Ra = np.std(one_series) if np.std(one_series) else 1e-6
        max_dim = n // time_delay - 1  # make sure at least 2 space points
        max_dim = min(max_dim, 5)  # control the upper limit to 5
        fnn = np.zeros(max_dim - 2)
        for dim in range(2, max_dim):
            space_points = n - dim * time_delay // self.sampling_interval  # make sure index d+1 exists
            y = self._psr_single_series(one_series, dim, time_delay)[:-time_delay, :]
            y0 = np.repeat(y, space_points, axis=0)
            y1 = np.tile(y, (space_points, 1))
            dist = np.sqrt(np.sum((y1 - y0) ** 2, axis=1))
            dist_with_matrix = np.reshape(dist, (space_points, space_points), order='C')
            sorted_dist, sorted_idx = np.sort(dist_with_matrix), np.argsort(dist_with_matrix)
            nearest_dist, nearest_idx = sorted_dist[:, 1], sorted_idx[:, 1]
            ref_idx = np.arange(space_points)
            D = np.abs(one_series[ref_idx + dim * time_delay] - one_series[nearest_idx + dim * time_delay])
            R = np.sqrt(D ** 2 + nearest_dist ** 2)
            nearest_dist[nearest_dist == 0] = 1e-6
            condition1 = D / nearest_dist
            condition2 = R / Ra
            fnn_counts = np.sum(np.logical_or(condition1 > rtol, condition2 > atol))
            fnn[dim - 2] = fnn_counts / space_points
        return np.argmin(fnn) + 2

    def _psr_single_series(self, one_series, dim, time_delay):
        n = one_series.shape[0]
        space_points = n - (dim - 1) * time_delay // self.sampling_interval
        res = np.zeros((space_points, dim))
        for i in range(space_points):
            res[i, :] = one_series[i: i + (dim - 1) * time_delay + 1:time_delay]
        return res

    @staticmethod
    def _calc_sup_norm(a):
        size = a.shape[0]
        # try to speedup the calculation
        tmp1 = np.repeat(a, size, axis=0)
        tmp2 = np.tile(a, (size, 1))
        diff = np.abs(tmp1 - tmp2)
        sup_norm = np.max(diff, axis=1)
        sup_norm_in_matrix = np.reshape(sup_norm, (size, size), order='C')
        upper_tri_idx = np.triu_indices(size)
        return sup_norm_in_matrix[upper_tri_idx]


class AmplitudesBasedGraph(BaseGraph):
    """
        constructing 2-d graphs from 1-d time series by grouping amplitude of each time sample, different amplitude groups
        will represent different nodes in the resulting graph, then we calculate moment vectors (contains k statistical moments)
        for each amplitude group, and use similarity among those moment vectors to reflect the connectivity among nodes in
        resulting group
    """

    def __init__(self,
                 num_of_amp_groups=10,
                 num_of_moments=4,
                 type='corr',
                 to_unweighted=True
                 ):
        """
            :param num_of_amp_groups: number of divided amplitude groups (equally divided between min and max)
            :param num_of_moments: number of the first moments used to construct moments vector
            :param type: different measurements to calculate connectivity among nodes (similarity embedded vectors);
                                'dist' -> 2-norm, 'corr' -> pearson correlation coefficient
            :param to_unweighted: save thresholds or not
        """
        self.number_of_nodes = num_of_amp_groups
        self.num_of_moments = num_of_moments
        self.to_unweighted = to_unweighted

        BaseGraph.__init__(self, type=type)

    def generate(self, one_time_series):
        """
        the amplitude embedding process
        """
        amp_groups = self._amplitude_partition(one_time_series)
        amp_moments = self._calc_moments(amp_groups)
        weighted_adj = self.measurement_func(amp_moments, amp_moments)

        w_to_unw_threshold = None
        if self.to_unweighted:
            w_to_unw_threshold = self.calc_best_conversion_threshold(weighted_adj, self.number_of_nodes)
        return weighted_adj, w_to_unw_threshold

    def _amplitude_partition(self, one_series):
        """
        this functionality will return the resulting vectors (groups) after amplitude partition
        """
        res = collections.defaultdict(list)
        min_amp, max_amp = np.min(one_series), np.max(one_series)
        if min_amp == max_amp:
            print('[INFO] cannot do amplitude partition !')
            return
        bin_width = (max_amp - min_amp) / self.number_of_nodes
        levels = [min_amp + i * bin_width for i in range(self.number_of_nodes)]
        for amp in one_series:
            index = bisect_left(levels, amp)  # boundary value belongs to previous bin
            index -= 1
            if index == 0:
                index += 1
            res[index].append(amp)
        res = list(res.values())
        return res

    def _calc_moments(self, lists):
        """
        generate moments vector for each amplitude group
        """
        res = np.zeros((self.number_of_nodes, self.num_of_moments))
        for i in range(len(lists)):
            if not lists[i]:
                continue
            res[i, 0] = np.mean(lists[i])
            for m in range(2, self.num_of_moments + 1):
                res[i, m - 1] = moment(lists[i], moment=m)
        min_res = np.min(res, axis=0)
        max_res = np.max(res, axis=0)
        res = (res - min_res) / (max_res - min_res)  # normalization
        return res


class GeneralisedRDPG:
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
