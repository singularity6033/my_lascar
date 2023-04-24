import collections
import gc
from bisect import bisect_left
from math import floor, log
from tqdm import tqdm
import numpy as np
from scipy.stats import norm, bernoulli, chi2, ks_2samp, cramervonmises_2samp, pearsonr, rv_histogram
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import KMeans
from TracyWidom import TracyWidom
from scipy.io import loadmat

gamma_diff_save = list()
miu_diff_save, rd_save, epsilon_save = [list(), list()], [list(), list()], [list(), list()]


class TwoSamplesGraphTesting:
    """
    Ghoshdastidar, Debarghya, and Ulrike Von Luxburg. "Practical methods for graph two-sample testing."
    Advances in Neural Information Processing Systems 31 (2018).
    """

    def __init__(self, r=3):
        """
        :param r: the number of communities (or rank) r only for approximation of P and Q used in the Tracy-Widom test
                  noted that the power of the test is not sensitive to the choice of r
        """
        self.r = r

    @staticmethod
    def mLarge_Testing(graph_sample_a, graph_sample_b, size_a, size_b, num_of_nodes):
        # it is a type of chi-square test statistic
        mean_grdpg_r, mean_grdpg_f = np.mean(graph_sample_a, axis=0), np.mean(graph_sample_b, axis=0)
        var_grdpg_r, var_grdpg_f = np.var(graph_sample_a, axis=0), np.var(graph_sample_b, axis=0)

        nominator = 0
        denominator = 0
        for i in range(num_of_nodes):
            for j in range(i + 1, num_of_nodes):
                term1 = (mean_grdpg_r[i][j] - mean_grdpg_f[i][j]) ** 2
                term2 = (var_grdpg_r[i][j] / size_a + var_grdpg_f[i][j] / size_b)
                if term1 != 0 and term2 == 0:
                    return 0.0
                elif term2 == 0:
                    continue
                nominator += term1
                denominator += term2
        if not denominator:
            return 0.0
        test_statistic = nominator / denominator
        test_statistic = np.nan_to_num(test_statistic)
        # Even though it evaluates the upper tail area, the chi-square test is regarded as a two-tailed test (non-directional)
        p_value = 1 - chi2.cdf(test_statistic, num_of_nodes * (num_of_nodes - 1) / 2)
        return p_value

    @staticmethod
    def mSmall_Testing(graph_sample_a, graph_sample_b, size_a, size_b, num_of_nodes):
        # proposed normality based test (Asymp-Normal)
        # https://github.com/gdebarghya/Network-TwoSampleTesting/blob/master/codes/NormalityTest.m
        p_value = 1.0
        m1 = floor(0.5 * min(size_a, size_b))
        term1_1 = np.sum(graph_sample_a[:m1, :, :] - graph_sample_b[:m1, :, :], axis=0)
        term1_2 = np.sum(graph_sample_a[m1:, :, :] - graph_sample_b[m1:, :, :], axis=0)
        term2_1 = np.sum(graph_sample_a[:m1, :, :] + graph_sample_b[:m1, :, :], axis=0)
        term2_2 = np.sum(graph_sample_a[m1:, :, :] + graph_sample_b[m1:, :, :], axis=0)

        term1_1 = np.triu(term1_1, 1)
        term1_2 = np.triu(term1_2, 1)
        term2_1 = np.triu(term2_1, 1)
        term2_2 = np.triu(term2_2, 1)
        nominator = np.sum(term1_1 * term1_2)
        denominator = np.sum(term2_1 * term2_2)

        if denominator:
            test_statistic = nominator / np.sqrt(denominator)
            p_value = 2 * norm.cdf(-abs(test_statistic))
        return p_value

    def mOne_Testing(self, graph_sample_a, graph_sample_b, num_of_nodes):
        # it is a type of Tracy-Widom test statistic
        # rewrite based on https://github.com/gdebarghya/Network-TwoSampleTesting/blob/master/codes/TracyWidomTest.m
        c = graph_sample_a - graph_sample_b
        idx = self._spectral_clustering(graph_sample_a, graph_sample_b)
        for i in range(self.r):
            for j in range(self.r):
                if i == j:
                    temp = graph_sample_a[np.ix_(i == idx, j == idx)]
                    # takes into account the diagonal is zero
                    Pij = 0 if temp.shape[0] <= 1 else np.sum(temp) / (temp.shape[0] * temp.shape[0] - 1)
                    temp = graph_sample_b[np.ix_(i == idx, j == idx)]
                    Qij = 0 if temp.shape[0] <= 1 else np.sum(temp) / (temp.shape[0] * temp.shape[0] - 1)
                    # continue
                else:
                    temp = graph_sample_a[np.ix_(i == idx, j == idx)]
                    Pij = np.mean(temp)
                    temp = graph_sample_b[np.ix_(i == idx, j == idx)]
                    Qij = np.mean(temp)
                denominator = np.sqrt((num_of_nodes - 1) * (Pij * (1 - Pij) + Qij * (1 - Qij)))
                if denominator == 0:
                    denominator = 1e-5
                c[np.ix_(i == idx, j == idx)] = c[np.ix_(i == idx, j == idx)] / denominator
        c = np.nan_to_num(c)
        u, s, et = np.linalg.svd(c)
        # using the largest singular value
        test_statistic = num_of_nodes ** (2 / 3) * (s[0] - 2)
        p_value = min(1, 2 * self._calc_tw_p(test_statistic))
        return p_value

    @staticmethod
    def _calc_tw_p(statistic):
        tw_data = loadmat('./Lib_SCA/lascar/engine/TW_beta1_CDF.mat')
        tw_cdf, tw_arg = tw_data['TW_CDF'], tw_data['TW_arg']
        ind = int(max(1, np.sum(tw_arg <= statistic)))
        p = 1 - tw_cdf[0, ind - 1]
        return p

    def _spectral_clustering(self, a, b):
        """
        spectral clustering to find common block structure
        a, b are two graphs, and r is a scalar (specifying number of communities)
        """
        c = (a + b) / 2
        d = np.sum(c, axis=1, keepdims=True)
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


class GraphCommunityBasedTesting:
    """
        Li, Yezheng, and Hongzhe Li. "Two-sample test of community memberships of weighted stochastic block models."
        arXiv preprint arXiv:1811.12593 (2018).
    """

    def __init__(self, graph_a, graph_b, number_of_node, k=3):
        """
        :param graph_a, graph_b: two independent graphs with the same group of n nodes and each is generated from a weighted SBM
        :param number_of_node: number of nodes in graph a and b (same)
        :param k: the number of communities (or rank or clusters)
        """
        self.graph_a = graph_a
        self.graph_b = graph_b
        self.number_of_node = number_of_node
        self.k = k

    def two_samples_community_memberships_testing(self):
        # labels = self._spectral_clustering_avg(self.graph_a, self.graph_b)
        tmp_ga, tmp_gb = 1 - self.graph_a, 1 - self.graph_b
        # lap_a = self.BaseGraph.calc_graph_laplacian_matrix(tmp_ga)
        # lap_b = self.BaseGraph.calc_graph_laplacian_matrix(tmp_gb)
        labels_a, labels_b = self._spectral_clustering_single(tmp_ga), self._spectral_clustering_single(tmp_gb)

        pa, qa, pb, qb = list(), list(), list(), list()  # intra and inter community distribution for two graphs

        for i in range(self.k):
            for j in range(i, self.k):
                if i == j:
                    # exclude diagonal and symmetric elements in this case
                    intra_a = self.graph_a[np.ix_(i == labels_a, j == labels_a)]
                    intra_b = self.graph_b[np.ix_(i == labels_b, j == labels_b)]
                    upper_tri_idx_a = np.triu_indices(intra_a.shape[0], 1)
                    upper_tri_idx_b = np.triu_indices(intra_b.shape[0], 1)
                    pa += intra_a[upper_tri_idx_a].tolist()
                    pb += intra_b[upper_tri_idx_b].tolist()
                else:
                    inter_a = self.graph_a[np.ix_(i == labels_a, j == labels_a)]
                    inter_b = self.graph_b[np.ix_(i == labels_b, j == labels_b)]
                    qa += inter_a.flatten().tolist()
                    qb += inter_b.flatten().tolist()

        pa, pb, qa, qb = np.array(pa), np.array(pb), np.array(qa), np.array(qb)
        pa0 = np.sum(pa == 0) / pa.shape[0]
        pb0 = np.sum(pb == 0) / pb.shape[0]
        qa0 = np.sum(qa == 0) / qa.shape[0]
        qb0 = np.sum(qb == 0) / qb.shape[0]
        pa, pb, qa, qb = pa[pa != 0], pb[pb != 0], qa[qa != 0], qb[qb != 0]

        # hist_pa, hist_qa = np.histogram(pa, bins=100), np.histogram(qa, bins=100)
        # hist_pb, hist_qb = np.histogram(pb, bins=100), np.histogram(qb, bins=100)
        # distribution_pa, distribution_qa = rv_histogram(hist_pa), rv_histogram(hist_qa)
        # distribution_pb, distribution_qb = rv_histogram(hist_pb), rv_histogram(hist_qb)
        #
        # miu_pa, miu_qa = distribution_pa.mean(), distribution_qa.mean()
        # miu_pb, miu_qb = distribution_pb.mean(), distribution_qb.mean()
        # sigma_pa, sigma_qa = distribution_pa.var(), distribution_qa.var()
        # sigma_pb, sigma_qb = distribution_pb.var(), distribution_qb.var()

        miu_pa, miu_pb = (1 - pa0) * np.mean(pa), (1 - pb0) * np.mean(pb)
        miu_qa, miu_qb = (1 - qa0) * np.mean(qa), (1 - qb0) * np.mean(qb)
        sigma_pa, sigma_pb = np.var(pa) * (1 - pa0) ** 2, np.var(pb) * (1 - pb0) ** 2
        sigma_qa, sigma_qb = np.var(qa) * (1 - qa0) ** 2, np.var(qb) * (1 - qa0) ** 2

        # assumption 3
        miu_diff_save[0].append(miu_pa - miu_qa)
        miu_diff_save[1].append(miu_pb - miu_qb)
        epsilon_save[0].append(log(miu_qa, self.number_of_node) + 0.5)
        epsilon_save[1].append(log(miu_qb, self.number_of_node) + 0.5)
        rd_a, rd_b = self._calc_renyi_divergence(pa, qa), self._calc_renyi_divergence(pb, qb)
        rd_save[0].append(self.number_of_node * rd_a / (np.log(self.number_of_node) * self.k))
        rd_save[1].append(self.number_of_node * rd_b / (np.log(self.number_of_node) * self.k))

        # assumption 4
        gamma_diff_save.append(abs(miu_pa / miu_pb - miu_qa / miu_qb))

        gamma = (miu_pa / miu_pb + miu_qa / miu_qb) / 2  # not so sure

        ua, sa, eta = np.linalg.svd(self.graph_a)
        ub, sb, etb = np.linalg.svd(self.graph_b)
        va, vb = ua[:, :self.k], ub[:, :self.k]  # num of nodes * k
        sb_k = np.diag(sb[:self.k])
        pt, _ = orthogonal_procrustes(va, vb)  # procrustes transformation
        # pt = va.T @ vb

        test_statistic = np.linalg.norm((va @ pt - vb) @ sb_k) ** 2 / (self.k * self.number_of_node)
        # test_statistic = np.linalg.norm((gamma * self.graph_a - self.graph_b) @ vb) ** 2 / (self.k * self.number_of_node)

        tmp_p, tmp_q = sigma_pa * gamma ** 2 + sigma_pb, sigma_qa * gamma ** 2 + sigma_qb
        miu = tmp_q + (gamma * sigma_pa + sigma_pb - tmp_q) / self.k
        sigma = (2 / (self.number_of_node * self.k)) * (tmp_q ** 2 + (tmp_p ** 2 - tmp_q ** 2) / self.k)
        normalized_ts = (test_statistic - miu) / np.sqrt(sigma)
        # xxx = 9 * np.log(1262) ** 4 / (1262 ** 2)

        p_value = 2 * (1 - norm.cdf(normalized_ts)) if normalized_ts >= 0 else 2 * norm.cdf(normalized_ts)
        return p_value

    @staticmethod
    def _calc_renyi_divergence(p, q):
        """
        calculate renyi divergence of given p and q (with order of 1 / 2)
        """
        # estimate distributions
        num_bins = 100
        min_b, max_b = min(np.min(p), np.min(q)), max(np.max(p), np.max(q))
        hist_p, _ = np.histogram(p, bins=num_bins, range=(min_b, max_b))
        hist_q, _ = np.histogram(q, bins=num_bins, range=(min_b, max_b))
        pdf_p, pdf_q = hist_p / np.sum(hist_p), hist_q / np.sum(hist_q)

        # discrete summation
        rd = -2 * np.log(np.sum(np.sqrt(pdf_p * pdf_q)))
        return rd

    def _community_estimation(self, l=5, beta=2):
        """
        algorithms used in Xu, Min et al. “Optimal rates for community estimation in the weighted stochastic block model.”
        param l: l labels used to convert SMB to Labeled-SBM
        param beta: cluster-imbalance constant
        """
        a = self.graph_a
        # algorithm 1: transformation and discretization
        bin_width = 1 / l
        bin_edges = [i * bin_width for i in range(l)]
        a_l = np.searchsorted(bin_edges, a, side='left')

        # algorithm 2: add noise
        p = 2 * (l + 1) / self.number_of_node
        noise_idx = 0
        for i in range(self.number_of_node):
            k = bernoulli.rvs(p, size=self.number_of_node - i - 1)
            for j in range(i + 1, self.number_of_node):
                if k[j - i - 1] == 1:
                    random_label = np.random.randint(0, l + 1)
                    a_l[i, j] = random_label
                    a_l[j, i] = a_l[i, j]
                noise_idx += 1

        # algorithm 3: initialization
        clustering_set = np.zeros((self.number_of_node, self.number_of_node - 1))
        v = np.array([list(range(self.number_of_node)) for _ in range(self.number_of_node)])
        v_tmp = np.ndarray.flatten(v)
        v_tmp = np.delete(v_tmp, range(0, len(v_tmp), len(v) + 1), 0)
        v_set = v_tmp.reshape(len(v), len(v) - 1)
        Il = np.zeros(l)
        ## stage 1
        miu = 4 * beta
        d_bar = np.sum(np.sum(a_l, axis=0) - np.diag(a_l)) / self.number_of_node
        tau = 40 * self.k * d_bar
        for i in tqdm(range(1, l + 1)):
            a_li = a_l.copy()
            a_li[a_li == i], a_li[a_li != i] = 1, 0
            labels = self._spectral_clustering(a_li, tau, miu)
            pl, ql = list(), list()
            for ki in range(self.k):
                for kj in range(ki, self.k):
                    if ki == kj:
                        intra = a_li[np.ix_(ki == labels, kj == labels)]
                        upper_tri_idx_a = np.triu_indices(intra.shape[0], 1)
                        pl += intra[upper_tri_idx_a].tolist()
                    else:
                        inter = a_li[np.ix_(ki == labels, kj == labels)]
                        ql += inter.flatten().tolist()
            pl, ql = np.mean(pl), np.mean(ql)
            Il[l - 1] = (((pl - ql) ** 2) / max(pl, ql))
        ## stage 2
        lx = np.argsort(Il)[-1] + 1
        a_lx = a_l.copy()
        a_lx[a_lx == lx], a_lx[a_lx != lx] = 1, 0
        for ui in range(self.number_of_node):
            a_lx_u = np.delete(a_lx, ui, 0)
            a_lx_u = np.delete(a_lx_u, ui, 1)
            clustering_set[ui, :] = self._spectral_clustering(a_lx_u, tau, miu)

        # algorithm 5: refinement
        clusters_hat = np.zeros(self.number_of_node)
        for uj in range(self.number_of_node):
            labels = clustering_set[uj, :]
            vj = v_set[uj, :]
            log_pql = np.zeros(l + 1)
            for i in range(l + 1):
                a_li = a_l.copy()
                a_li[a_li == i], a_li[a_li != i] = 1, 0
                pl, ql = list(), list()
                for ki in range(self.k):
                    for kj in range(ki, self.k):
                        if ki == kj:
                            intra = a_li[np.ix_(vj[ki == labels], vj[kj == labels])]
                            upper_tri_idx_a = np.triu_indices(intra.shape[0], 1)
                            pl += intra[upper_tri_idx_a].tolist()
                        else:
                            inter = a_li[np.ix_(vj[ki == labels], vj[kj == labels])]
                            ql += inter.flatten().tolist()
                pl, ql = np.mean(pl), np.mean(ql)
                if ql == 0:
                    ql = 1
                log_pql[i] = np.log(pl / ql)

            k_selected = np.zeros(self.k)
            for kk in range(self.k):
                vk = vj[kk == labels]
                lv = a_l[uj, vk]
                for kl in range(l + 1):
                    k_selected[vk] += np.count_nonzero(lv == kl) * log_pql[kl]
            clusters_hat[uj] = np.argsort(k_selected)[-1]

        ## consensus stage
        clusters_hat_res = np.zeros(self.number_of_node)
        clusters_hat_res[0] = clusters_hat[0]
        for uc in range(1, self.number_of_node):
            k_selected = np.zeros(self.k)
            ku = clusters_hat[uc]
            for kc in range(self.k):
                v1 = np.where(clusters_hat == kc)
                v2 = np.where(clusters_hat == ku)
                k_selected[kc] = len(np.intersect1d(v1, v2))
            clusters_hat_res[uc] = np.argsort(k_selected)[-1]
        return clusters_hat_res

    def _spectral_clustering(self, a, tau, miu):
        # algorithm 4
        # an unweighted network a with columns {au}, trim threshold tau , tuning parameter miu
        degree_a = np.sum(a, axis=0) - np.diag(a)
        idx = degree_a >= tau
        ta = a.copy()
        ta[idx, :], ta[:, idx] = 0, 0
        u, s, et = np.linalg.svd(ta)
        a_hat = u[:, :self.k] @ np.diag(s[:self.k]) @ et[:self.k, :]

        tmp1 = np.repeat(a_hat, self.number_of_node, axis=1)
        tmp2 = np.tile(a_hat, (1, self.number_of_node))
        l2 = np.sqrt(np.sum((tmp1 - tmp2) ** 2, axis=0))
        l2_with_matrix = np.reshape(l2, (self.number_of_node, self.number_of_node), order='C')
        sorted_l2 = np.sort(l2_with_matrix)
        d_idx = int(self.number_of_node / (miu * self.k))
        d = sorted_l2[:, d_idx]

        S = list()
        sorted_idx = np.argsort(d)[0]
        S.append(sorted_idx)

        quantile = 1 - 1 / (miu * self.k)
        d_q = np.quantile(d, quantile)
        u_node = []
        for node_idx in range(self.number_of_node):
            if d[node_idx] <= d_q:
                u_node.append(node_idx)
        a_hat_u = a_hat[:, u_node]
        for i in range(1, self.k):
            a_hat_v = a_hat[:, S]
            tmp_u = np.repeat(a_hat_u, len(S), axis=1)
            tmp_v = np.tile(a_hat_v, (1, len(u_node)))
            l2_uv = np.sqrt(np.sum((tmp_u - tmp_v) ** 2, axis=0))
            l2_uv_with_matrix = np.reshape(l2_uv, (len(u_node), len(S)), order='C')
            sorted_l2_uv = np.sort(l2_uv_with_matrix)
            ui = np.argsort(sorted_l2_uv[:, 0])[-1]
            S.append(ui)

        a_hat_s = a_hat[:, S]
        res_u = np.repeat(a_hat, len(S), axis=1)
        res_s = np.tile(a_hat_s, (1, self.number_of_node))
        l2_us = np.sqrt(np.sum((res_u - res_s) ** 2, axis=0))
        l2_us_with_matrix = np.reshape(l2_us, (self.number_of_node, len(S)), order='C')
        sorted_l2_us = np.argsort(l2_us_with_matrix)
        labels = sorted_l2_us[:, 0]
        return labels

    def _spectral_clustering_avg(self, a, b):
        """
        spectral clustering to find common block structure
        a, b are two graphs, and r is a scalar (specifying number of communities)
        """
        c = (a + b) / 2
        d = np.sum(c, axis=1, keepdims=True)
        d[d == 0] = 1
        d = 1 / np.sqrt(d)
        c = c * np.dot(d, d.T)
        u, s, et = np.linalg.svd(c)
        # using r largest dominant singular vectors
        e = np.array(et[:self.k, :], ndmin=2).T
        norm_e = np.array(np.sqrt(np.sum(e ** 2, axis=0)), ndmin=2)
        norm_e[norm_e == 0] = 1
        e = e / norm_e
        km = KMeans(n_clusters=self.k, random_state=0).fit(e)
        return km.labels_

    def _spectral_clustering_single(self, g):
        d = np.sum(g, axis=1, keepdims=True)
        d[d == 0] = 1
        d = 1 / np.sqrt(d)
        g = g * np.dot(d, d.T)
        u, s, et = np.linalg.svd(g)
        # using r largest dominant singular vectors
        e = np.array(et[:self.k, :], ndmin=2).T
        norm_e = np.array(np.sqrt(np.sum(e ** 2, axis=0)), ndmin=2)
        norm_e[norm_e == 0] = 1
        e = e / norm_e
        km = KMeans(n_clusters=self.k, random_state=0).fit(e)
        return km.labels_


class Chi2Test:
    def __init__(self, d0, d1, num_bins):
        self.d0 = d0
        self.d1 = d1
        self.num_bins = num_bins

    def output(self):
        min_b, max_b = min(np.min(self.d0), np.min(self.d1)), max(np.max(self.d0), np.max(self.d1))
        hist_d0, _ = np.histogram(self.d0, bins=self.num_bins, range=(min_b, max_b))
        hist_d1, _ = np.histogram(self.d1, bins=self.num_bins, range=(min_b, max_b))
        hist_d0, hist_d1 = np.array(hist_d0, ndmin=2), np.array(hist_d1, ndmin=2)

        chi2_table = np.concatenate((hist_d0, hist_d1), axis=0)
        p_value = self._calc_chi2test(chi2_table)
        if p_value == 0:
            # minimum value in float
            p_value = np.finfo(float).tiny
        return p_value

    @staticmethod
    def histogram_count(data, b_arr):
        hist_count = np.zeros(b_arr.shape[0] - 1)
        for data_i in data:
            index = bisect_left(b_arr, data_i)
            if not index == 0:
                index -= 1
            hist_count[index] += 1
        return hist_count

    @staticmethod
    def _calc_chi2test(chi2_table):
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
