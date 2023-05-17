import numpy as np
from scipy.stats import ks_2samp
import netcomp as nt

gamma_diff_save = list()
miu_diff_save, rd_save, epsilon_save = [list(), list()], [list(), list()], [list(), list()]


class GraphDistance:

    def __init__(self, mode='edit_dist'):
        if mode == 'spectral_dist':
            self.graph_distance = self.spectral_dist
        elif mode == 'edit_dist':
            self.graph_distance = self.edit_dist
        elif mode == 'vertex_edge_dist':
            self.graph_distance = self.vertex_edge_dist
        elif mode == 'deltacon0_dist':
            self.graph_distance = self.deltacon0_dist
        elif mode == 'resistance_dist':
            self.graph_distance = self.resistance_dist
        elif mode == 'corr_dist':
            self.graph_distance = self.corr_dist
        elif mode == 'chi2_dist':
            self.graph_distance = self.chi2_score

    @staticmethod
    def spectral_dist(a, b, params):
        ki, pi, kdi = params['k'], params['p'], params['kind']
        if kdi == 'adjacency':
            return nt.lambda_dist(a, b, k=ki, p=pi, kind='adjacency')
        elif kdi == 'laplacian':
            return nt.lambda_dist(a, b, k=ki, p=pi, kind='laplacian')
        elif kdi == 'laplacian_norm':
            return nt.lambda_dist(a, b, k=ki, p=pi, kind='laplacian_norm')

    @staticmethod
    def edit_dist(a, b):
        dist = np.abs((a - b)).sum() / 2
        return dist

    @staticmethod
    def vertex_edge_dist(a, b):
        return nt.vertex_edge_distance(a, b)

    @staticmethod
    def deltacon0_dist(a, b):
        return nt.deltacon0(a, b)

    @staticmethod
    def resistance_dist(a, b):
        return nt.resistance_distance(a, b)

    @staticmethod
    def corr_dist(a, b):
        m = a.shape[0]
        upper_tri_idx = np.triu_indices(m, 1)
        ap = a[upper_tri_idx]
        bp = b[upper_tri_idx]
        from scipy.stats.stats import pearsonr
        c = abs(pearsonr(ap, bp)[0])
        # can convert to dist by (1-c)/c
        return c

    @staticmethod
    def chi2_score(data, num_bins):
        min_data, max_data = np.min(data), np.max(data)
        bin_width = (max_data - min_data) / num_bins
        hist_ranges = np.array([min_data + i * bin_width for i in range(num_bins)])
        hist_counts = np.zeros(hist_ranges.shape[0] * 9)

        data_flatten = data.flatten(order='C')
        count_index = np.searchsorted(hist_ranges, data_flatten, side='left')
        count_index[data_flatten != hist_ranges[0]] -= 1

        sample_size = data.shape[1]
        sample_idx = 0
        for i, c_idx in enumerate(count_index):
            if i > 0 and i % sample_size == 0:
                sample_idx += 1
            hist_counts[c_idx + sample_idx * num_bins] += 1

        chi2_table = np.reshape(hist_counts, (9, -1), order='C')
        n = np.sum(chi2_table)
        col_sum = np.sum(chi2_table, axis=0, keepdims=True)
        row_sum = np.sum(chi2_table, axis=1, keepdims=True)
        expected_freq = np.dot(row_sum, col_sum) / n
        tmp1 = (chi2_table - expected_freq) ** 2
        tmp2 = np.divide(tmp1, expected_freq, out=np.zeros_like(tmp1), where=expected_freq != 0)
        chi_score = np.sum(tmp2)
        return chi_score
