import collections
import tracemalloc
from bisect import bisect_left
from collections import Counter

import numpy as np
from numpy import e
from scipy import integrate
from sklearn.neighbors import KernelDensity
from scipy.stats import binom, rv_histogram, norm
from tqdm import tqdm
from sys import getsizeof

from . import GuessEngine


class CMI_Engine_By_Histogram(GuessEngine):
    """
    CMI_Engine_By_Histogram is a specialized Engine which is used to calculate the mutual information of the leakage as well as
    the statistical test the implementation refers to the paper related to the continuous mutual information (CMI)
    proposed in Chothia, Tom, and Apratim Guha. "A statistical test for information leaks using continuous mutual
    information." 2011 IEEE 24th Computer Security Foundations Symposium. IEEE, 2011.

    the method used to estimate pdfs involved in the calculation of the cmi is the histogram-based estimation, which is
    implemented incrementally
    """

    def __init__(self,
                 name,
                 selection_function,
                 guess_range,
                 num_bins=5,
                 hist_boundary=None,
                 num_shuffles=100,
                 solution=-1,
                 jit=True):
        """
        MiEngine
        :param name: name of the engine
        :param selection_function: takes a value and a guess_guess as input, returns a modelisation of the leakage for this (value/guess).
        :param num_bins: number of bins used in initialization of histogram estimation
        :param hist_boundary: pre-defined hist boundary, if == None, the boundary is based on data of the first batch
        :param guess_range: what are the values for the guess guess
        :param num_shuffles: random shuffle times used to obtain the reasonable statistical test value
        """
        self.num_bins = num_bins
        self.hist_boundary = hist_boundary
        bin_width = (hist_boundary[1] - hist_boundary[0]) / num_bins
        # levels for measurement y
        self.levels = [hist_boundary[0] + i * bin_width for i in range(num_bins)]
        self.levels.append(hist_boundary[1])
        # levels for secret x
        self.levels_sx = [0 + i * 1 for i in range(8)]
        self.levels_sx.append(8)

        self.num_shuffles = num_shuffles

        self.results = None
        GuessEngine.__init__(self, name, selection_function, guess_range, solution, jit)

    def _initialize(self):
        self._mutual_information = np.zeros((self._number_of_guesses,) + self._session.leakage_shape, np.double)
        self._p_value = np.zeros((self._number_of_guesses,) + self._session.leakage_shape, np.double)

        self.number_of_time_samples = self._mutual_information.shape[1]

        self.pdfs_for_pyx = np.zeros((self._number_of_guesses, self.num_shuffles + 1, self.number_of_time_samples, 9, self.num_bins))
        self.pdfs_for_px = np.zeros((self._number_of_guesses, 8))
        self.pdfs_for_py = np.zeros((self.number_of_time_samples, self.num_bins))

        # as size for y given x is unknown
        self.y_x = [[[collections.defaultdict(list)
                      for _ in range(self.number_of_time_samples)]
                     for __ in range(self.num_shuffles + 1)]
                    for ___ in range(self._number_of_guesses)]

        self._batch_count = 0

        self.size_in_memory += self._mutual_information.nbytes
        self.size_in_memory += self._p_value.nbytes
        self.size_in_memory += self.pdfs_for_pyx.nbytes
        self.size_in_memory += self.pdfs_for_px.nbytes
        self.size_in_memory += self.pdfs_for_py.nbytes

    def _histogram_estimation_px(self, data, key_guess_idx):
        for data_i in data:
            index = bisect_left(self.levels_sx, data_i) - 1 if not data_i == self.levels_sx[0] else 0
            self.pdfs_for_px[key_guess_idx, index] += 1

    def _histogram_estimation_py(self, data, time_sample_idx):
        for data_i in data:
            index = bisect_left(self.levels, data_i) - 1 if not data_i == self.levels[0] else 0
            self.pdfs_for_py[time_sample_idx, index] += 1

    def _histogram_estimation_p_yx(self, key_guess_idx, c_idx, time_sample_idx, y, secret_x_i, secret_x_i_set):
        """
            1. estimate the histogram of p_yx for current batch
            2. store y_x for later processing
        """
        for secret_x_val in secret_x_i_set:
            secret_index = np.where(secret_x_i == secret_x_val)
            y_x = y[secret_index]
            for y_x_i in y_x:
                index = bisect_left(self.levels, y_x_i) - 1 if not y_x_i == self.levels[0] else 0
                self.pdfs_for_pyx[key_guess_idx, c_idx, time_sample_idx, secret_x_val, index] += 1

    def _store_yx(self, key_guess_idx, c_idx, time_sample_idx, y, secret_x_i, secret_x_i_set):
        for secret_x_val in secret_x_i_set:
            secret_index = np.where(secret_x_i == secret_x_val)
            y_x = y[secret_index].flatten().tolist()
            self.y_x[key_guess_idx][c_idx][time_sample_idx][secret_x_val] += y_x

    def _cal_integration_term(self, p_y, p_yx, yx):
        p_y_dist = rv_histogram((p_y, np.array(self.levels)))
        p_yx_dist = rv_histogram((p_yx, np.array(self.levels)))
        pyx = p_yx_dist.pdf(yx)
        sum_term = p_y_dist.pdf(yx)
        # extreme value processing
        sum_term_mask = sum_term == 0.0
        sum_term[sum_term_mask] = 1.0
        log_term = pyx / sum_term
        log_term_mask = log_term == 0.0
        log_term[log_term_mask] = 1.0
        return pyx * np.log(log_term)

    def _cal_mutual_information(self, p_x, p_y, p_yx, yx_total):
        # number of different x
        len_x = len(p_yx)
        cmi = 0
        p_x_dist = rv_histogram((p_x, np.array(self.levels_sx)))
        # calculate eqn (4)
        for xi in range(len_x):
            # integration part
            yx = np.unique(yx_total[xi])
            integrate_term = self._cal_integration_term(p_y, p_yx[xi], yx)
            integrate_res = integrate_term if yx.shape[0] < 2 else integrate.trapezoid(integrate_term, yx)
            cmi += p_x_dist.pdf(xi) * integrate_res
        return cmi

    def _update(self, batch):
        """
        1. we use incremental histogram estimation to estimate the pdf of p(y|x) for each
        key guess, test, time sample and secret x value (hamming)
        2. we self-define a merge function to combine the histogram of previous batch and the histogram of current batch
        3. we assume the distribution of input secret x (hamming) is known as a binomial distribution with n=8, p=0.5
        """
        print('[INFO] Batch #', self._batch_count + 1)
        secret_x = self._mapfunction(self._guess_range, batch.values)  # batch_size * guess_range
        batch_leakages = batch.leakages

        # estimate the histogram of p_y for current batch
        for ti in range(self.number_of_time_samples):
            y = batch_leakages[:, ti]
            self._histogram_estimation_py(y, ti)

        for i in tqdm(range(self._number_of_guesses)):
            print('[INFO] Processing Key Guess #', i)
            secret_x_i = secret_x[:, i]
            secret_x_i_set = np.unique(secret_x_i)  # no repeated item

            # estimate the histogram of p_x for current key guess
            self._histogram_estimation_px(secret_x_i, i)

            for j in range(self.number_of_time_samples):
                secret_x_i_copy = np.copy(secret_x_i)
                y = np.array(batch_leakages[:, j], ndmin=2).T

                # estimate the histogram of p_y for current batch
                self._histogram_estimation_p_yx(i, 0, j, y, secret_x_i, secret_x_i_set)
                self._store_yx(i, 0, j, y, secret_x_i, secret_x_i_set)

                # statistical test
                for k in range(1, self.num_shuffles + 1):
                    np.random.shuffle(secret_x_i_copy)
                    self._histogram_estimation_p_yx(i, k, j, y, secret_x_i_copy, secret_x_i_set)
                    self._store_yx(i, k, j, y, secret_x_i_copy, secret_x_i_set)

        self._batch_count += 1

    def _finalize(self):
        print('[INFO] processing cmi calculation...')
        for i in tqdm(range(self._number_of_guesses)):
            p_x = self.pdfs_for_px[i]
            for j in range(self.number_of_time_samples):
                p_y = self.pdfs_for_py[j]
                p_yx = self.pdfs_for_pyx[i][0][j]
                yx = self.y_x[i][0][j]
                # calculate real cmi
                real_cmi = self._cal_mutual_information(p_x, p_y, p_yx, yx)
                self._mutual_information[i][j] = real_cmi

                # statistical test
                cmi_zero_leakages = np.zeros(self.num_shuffles)
                for k in range(1, self.num_shuffles + 1):
                    p_yx = self.pdfs_for_pyx[i][k][j]
                    yx = self.y_x[i][k][j]  # list
                    cmi_shuffle = self._cal_mutual_information(p_x, p_y, p_yx, yx)
                    cmi_zero_leakages[k - 1] = cmi_shuffle

                # theorem 1
                m = np.mean(cmi_zero_leakages)
                v = np.std(cmi_zero_leakages)
                p_value = 2 * norm.cdf(real_cmi, loc=m, scale=v) if real_cmi < m else 2 * (
                        1 - norm.cdf(real_cmi, loc=m, scale=v))
                self._p_value[i][j] = p_value
        results = (self._mutual_information, self._p_value)
        # self._clean()
        return results

    def _clean(self):
        import gc
        del self._mutual_information
        del self._p_value
        del self.number_of_time_samples
        del self.pdfs_for_pyx
        del self.y_x
        del self.pdfs_for_px
        del self.pdfs_for_py
        gc.collect()
        self.size_in_memory = 0


class CMI_Engine_By_KDE(GuessEngine):
    """
    CMI_Engine is a specialized Engine which is used to calculate the mutual information of the leakage as well as the
    statistical test
    the implementation refers to the paper related to the continuous mutual information (CMI) proposed in
    Chothia, Tom, and Apratim Guha. "A statistical test for information leaks using continuous mutual information."
    2011 IEEE 24th Computer Security Foundations Symposium. IEEE, 2011.

    the method used to estimate pdfs involved in the calculation of the cmi is the kernel density based estimation
    but this is not implemented on an incremental manner (calculate only once)
    """

    def __init__(self,
                 name,
                 selection_function,
                 guess_range,
                 kernel='epanechnikov',
                 contain_test=True,
                 num_shuffles=100,
                 solution=None,
                 jit=True):
        """
        MiEngine
        :param name: name of the engine
        :param selection_function: takes a value and a guess_guess as input, returns a modelisation of the leakage for this (value/guess).
        :param guess_range: what are the values for the guess guess
        :param kernel: kernel type used in the pdf estimation
        :param num_shuffles: testing times used to obtain the reasonable statistical value
        if contain_test is True it will return real mean continuous mutual information + test scores
        else it will only return real mean continuous mutual information
        """
        self.kernel = kernel
        self.contain_test = contain_test
        self.num_shuffles = num_shuffles
        GuessEngine.__init__(self, name, selection_function, guess_range, solution, jit)

    def _initialize(self):
        self._mutual_information = np.zeros((self._number_of_guesses,) + self._session.leakage_shape, np.double)
        self._mutual_information_ref = np.zeros((self._number_of_guesses,) + self._session.leakage_shape, np.double)
        self._p_value = np.zeros((self._number_of_guesses,) + self._session.leakage_shape, np.double)
        self._p_value_ref = np.zeros((self._number_of_guesses,) + self._session.leakage_shape, np.double)
        self._batch_count = np.zeros((self._number_of_guesses,) + self._session.leakage_shape, dtype=np.int32, )

        self.size_in_memory += self._mutual_information.nbytes
        self.size_in_memory += self._mutual_information_ref.nbytes
        self.size_in_memory += self._p_value.nbytes
        self.size_in_memory += self._p_value_ref.nbytes
        self.size_in_memory += self._batch_count.nbytes

    def _update(self, batch):
        """
        1. we use kernel estimation to estimate the pdf of p(y|x), kernel used in CMI is epanechnikov kernel and bandwidth
        can be set with general purpose (5) in the paper
        2. we assume the distribution of input secret x (hw model out) is known as a binomial distribution with n=8, p=0.5
        3. we use the equation 4 in the paper to calculate continuous mutual information
        """
        secret_x = self._mapfunction(self._guess_range, batch.values)  # batch_size * guess_range
        batch_leakages = batch.leakages
        batch_size = batch_leakages.shape[0]
        time_samples = batch_leakages.shape[1]
        # p_x (binomial distribution) - known
        p_x = binom.pmf(secret_x, n=8, p=0.5)
        for i in tqdm(range(self._number_of_guesses)):
            print('[INFO] Processing Key Guess', i)
            secret_x_i = secret_x[:, i]
            p_x_i = p_x[:, i]
            secret_x_i_set = np.unique(secret_x_i)  # no repeated item
            for j in tqdm(range(time_samples)):
                secret_x_i_copy = np.copy(secret_x_i)
                p_x_i_copy = np.copy(p_x_i)
                lk_total = np.array(batch_leakages[:, j], ndmin=2).T
                # bandwidth used in the kernel estimation
                bandwidth = 1.06 * np.sqrt(np.var(lk_total)) * batch_size ** (-1 / 5)
                # directly estimate p_y = sum(p_xi * p_yxi)
                p_y = KernelDensity(kernel=self.kernel, bandwidth=bandwidth).fit(lk_total)
                kde_set, cmi = self._cal_mutual_information(secret_x_i, secret_x_i_set, p_y, lk_total, bandwidth)
                cmi_ref = self._cal_mutual_information_reference(secret_x_i, batch_leakages[:, j])

                # mutual information statistic
                self._mutual_information[i][j] += cmi
                self._mutual_information_ref[i][j] += cmi_ref
                self._batch_count[i][j] += 1

                # statistical test
                cmi_zero_leakages = np.zeros(self.num_shuffles)
                cmi_zero_leakages_ref = np.zeros(self.num_shuffles)
                for k in range(self.num_shuffles):
                    np.random.shuffle(secret_x_i_copy)
                    cmi_shuffle = self._cal_cmi_statistical_test(secret_x_i_copy, secret_x_i_set, p_y, kde_set,
                                                                 lk_total)
                    cmi_shuffle_ref = self._cal_mutual_information_reference(secret_x_i_copy, batch_leakages[:, j])
                    cmi_zero_leakages[k] = cmi_shuffle
                    cmi_zero_leakages_ref[k] = cmi_shuffle_ref

                # theorem 1
                c1 = np.mean(cmi_zero_leakages)
                c2 = np.var(cmi_zero_leakages)
                m = c1 * np.log2(np.e) / (batch_size * bandwidth)
                v = np.sqrt(c2 * np.power(np.log2(np.e), 2) / ((batch_size ** 2) * bandwidth ** (-1)))
                c1_ref = np.mean(cmi_zero_leakages_ref)
                c2_ref = np.var(cmi_zero_leakages_ref)
                m_ref = c1_ref * np.log2(np.e) / (batch_size * bandwidth)
                v_ref = np.sqrt(c2_ref * np.power(np.log2(np.e), 2) / ((batch_size ** 2) * bandwidth ** (-1)))
                from scipy.stats import norm
                p_value = 2 * norm.cdf(cmi, loc=m, scale=v) if cmi < m else 2 * (1 - norm.cdf(cmi, loc=m, scale=v))
                p_value_ref = 2 * norm.cdf(cmi_ref, loc=m_ref, scale=v_ref) if cmi_ref < m_ref else \
                    2 * (1 - norm.cdf(cmi_ref, loc=m_ref, scale=v_ref))
                self._p_value[i][j] += p_value
                self._p_value_ref[i][j] += p_value_ref

    @staticmethod
    def _cal_integration_term(lk, p_yx, p_y):
        lk = np.array(lk, ndmin=2).T
        pyx = np.exp(p_yx.score_samples(lk))
        sum_term = np.exp(p_y.score_samples(lk))
        # extreme value processing
        sum_term_mask = sum_term == 0.0
        sum_term[sum_term_mask] = 1.0
        log_term = pyx / sum_term
        log_term_mask = log_term == 0.0
        log_term[log_term_mask] = 1.0
        return pyx * np.log(log_term)

    def _cal_mutual_information(self, x, x_set, p_y, lk, bandwidth):
        kde_set = []  # for each element in set, estimate corresponding pdf of yx
        lk_set = []
        # calculate every p_yx and corresponding yx
        for xi in x_set:
            index = np.where(x == xi)
            lk_set_i = lk[index]
            kde = KernelDensity(kernel=self.kernel, bandwidth=bandwidth).fit(lk_set_i)
            kde_set.append(kde)
            lk_set.append(lk_set_i)
        cmi = 0
        # calculate eqn (4)
        for i in range(x_set.shape[0]):
            # integration part
            lk_series = np.unique(lk_set[i])
            f_lk = self._cal_integration_term(lk_series, kde_set[i], p_y)
            integrate_term = integrate.trapezoid(f_lk, lk_series)
            cmi += binom.pmf(x_set[i], n=8, p=0.5) * integrate_term
        return kde_set, cmi

    @staticmethod
    def _cal_mutual_information_reference(x, y):
        """
        using sklearn.feature_selection.mutual_info_regression to calculation between discrete x and continuous y
        """
        import sklearn.feature_selection
        x = np.array(x, ndmin=2).T
        cmi_ref = sklearn.feature_selection.mutual_info_regression(x, y)
        return cmi_ref

    def _cal_cmi_statistical_test(self, x, x_set, p_y, kde_set, lk):
        lk_set = []
        # calculate related terms
        for xi in x_set:
            index = np.where(x == xi)
            lk_set_i = lk[index]
            lk_set.append(lk_set_i)
        cmi = 0
        # calculate eqn (4)
        for i in range(x_set.shape[0]):
            # integration part
            lk_series = np.unique(lk_set[i])
            f_lk = self._cal_integration_term(lk_series, kde_set[i], p_y)
            integrate_term = integrate.trapezoid(f_lk, lk_series)
            cmi += binom.pmf(x_set[i], n=8, p=0.5) * integrate_term
        return cmi

    def _finalize(self):
        mi = self._mutual_information / self._batch_count
        mi_ref = self._mutual_information_ref / self._batch_count
        pv = self._p_value / self._batch_count
        pv_ref = self._p_value_ref / self._batch_count
        return mi, mi_ref, pv, pv_ref

    def _clean(self):
        del self._mutual_information
        del self._mutual_information_ref
        del self._p_value
        del self._p_value_ref
        del self._batch_count
        self.size_in_memory = 0
