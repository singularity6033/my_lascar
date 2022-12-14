import gc

import numpy as np
from scipy.stats import norm, bernoulli, chi2, ks_2samp, cramervonmises_2samp
import sklearn.feature_selection as fs
from tqdm import tqdm

from . import PartitionerEngine, Phase_Space_Reconstruction_Graph


class GraphMIEngine(PartitionerEngine):
    """
        GraphMIEngine is a graph-based mi engine, used to calculate random graphs with known distribution (pdf of pmf)
        and form a cmi-like statistical inference to obtain p-value as well

        random graphs is a type of , obtained from a generalised RDPG construction with 'single' mode
    """

    def __init__(self, name, partition_function, time_delay, dim, sampling_interval=1, mi_mode='simple',
                 num_shuffles=50):
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
        self.mi_mode = mi_mode
        # if mi_mode == 'simple':
        #     self.mi_method = self._calc_graph_mutual_information_simple
        # elif mi_mode == 'sklearn':
        #     self.mi_method = self._calc_graph_mutual_information_sklearn
        # elif mi_mode == 'estimate':
        #     self.mi_method = self._calc_graph_mutual_information_estimate

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
        real_mi = eval('self._calc_graph_mutual_information_' + self.mi_mode)(graph_samples_r, graph_samples_f)

        # reference mi (same distribution) used in statistical testing (assuming a gaussian distribution)
        # both samples randomly drawn from a inhomogeneous ER random graph with each edge is a independent r.v.
        # follows a Bernoulli distribution (p=0.5)
        reference_mi = np.zeros(self.num_shuffles)
        # p = 0.5 * np.ones((m_r, self.number_of_nodes, self.number_of_nodes))
        for i in tqdm(range(self.num_shuffles)):
            np.random.shuffle(random_set)
            shuffled_graph_r = Phase_Space_Reconstruction_Graph(random_set, self.time_delay, self.dim,
                                                                self.sampling_interval)
            shuffled_graph_r.generate()
            shuffled_graph_samples_r = shuffled_graph_r.adj_matrix
            # ier_graph_a = bernoulli.rvs(p, size=(m_r, self.number_of_nodes, self.number_of_nodes))
            # ier_graph_b = bernoulli.rvs(p, size=(m_f, self.number_of_nodes, self.number_of_nodes))
            reference_mi[i] = eval('self._calc_graph_mutual_information_' + self.mi_mode)(graph_samples_r,
                                                                                          shuffled_graph_samples_r)

        # p value calculation
        m = np.mean(reference_mi)
        v = np.std(reference_mi)
        p_value = 2 * norm.cdf(real_mi, loc=m, scale=v) if real_mi < m else 2 * (1 - norm.cdf(real_mi, loc=m, scale=v))
        self._clean()
        return real_mi, p_value

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
        pass

    def _clean(self):
        del self._samples_by_partition
        del self._partition_count
        gc.collect()
        self.size_in_memory = 0