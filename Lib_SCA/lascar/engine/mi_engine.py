from bisect import bisect_left
from collections import Counter

import numpy as np
from numpy import e
from scipy import integrate
from sklearn.neighbors import KernelDensity
from scipy.stats import binom, rv_histogram, norm
from tqdm import tqdm

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
                 solution=None,
                 jit=True):
        """
        MiEngine
        :param name: name of the engine
        :param selection_function: takes a value and a guess_guess as input, returns a modelisation of the leakage for this (value/guess).
        :param num_bins: number of bins used in initialization of histogram estimation
                         1. if == 0, it will be in an 'auto' mode, it will calculate current hist based on
                         the current data and merge previous hist and current hist directly based on
                         https://stackoverflow.com/questions/47085662/merge-histograms-with-different-ranges
                         2. if > 0, it will update previous hist with current raw data with fixed bin sizes
        :param hist_boundary: pre-defined hist boundary, if == None, the boundary is based on data of the first batch
        :param guess_range: what are the values for the guess guess
        :param num_shuffles: testing times used to obtain the reasonable statistical value
        """
        self.num_bins = num_bins
        self.hist_boundary = hist_boundary
        self.num_shuffles = num_shuffles
        self.results = None
        GuessEngine.__init__(self, name, selection_function, guess_range, solution, jit)

    def _initialize(self):
        self.secret_x = None
        self.y_total = None

        self._mutual_information = np.zeros((self._number_of_guesses,) + self._session.leakage_shape, np.double)
        self._p_value = np.zeros((self._number_of_guesses,) + self._session.leakage_shape, np.double)

        self.number_of_time_samples = self._mutual_information.shape[1]

        # 4 dimensional list to store each pyx and yx for each key guess, test, time sample and secret x value (hamming)
        self.pdfs_for_pyx = [[[[None] * 9
                               for _ in range(self.number_of_time_samples)]
                              for __ in range(self.num_shuffles + 1)]
                             for ___ in range(self._number_of_guesses)]
        self.y_x = [[[[None] * 9
                      for _ in range(self.number_of_time_samples)]
                     for __ in range(self.num_shuffles + 1)]
                    for ___ in range(self._number_of_guesses)]

        # 3 dimensional list to store each px, py for each key guess, test, time sample
        self.pdfs_for_px = [None] * self._number_of_guesses
        self.pdfs_for_py = [None] * self.number_of_time_samples

        self._batch_count = 0

        self.size_in_memory += self._mutual_information.nbytes
        self.size_in_memory += self._p_value.nbytes

    def update_hist(self, prev_hist, cur_data):
        """
        this update_hist function directly update the previous histogram based on the current data
        it may involve padding operations
        """
        counter_dic = Counter(cur_data.flatten())
        min_data, max_data = min(counter_dic.keys()), max(counter_dic.keys())
        new_hist, new_bin_edges = self._update_histogram_range(prev_hist, min_data, max_data)
        for data_i in cur_data:
            index = bisect_left(new_bin_edges, data_i) - 1
            new_hist[index] += 1
        return new_hist, new_bin_edges

    @staticmethod
    def merge_hist(prev_hist, cur_hist):
        """
        this merge_hist function is based on two assumptions:
        1. Recovered values are represented by the start of the bin they belong to.
        2. The merge shall keep the highest bin resolution to avoid further loss of information
        and shall completely encompass the intervals of the children histograms.
        """

        def extract_vals(hist):
            # Recover values based on assumption 1.
            values = [[y] * x for x, y in zip(hist[0], hist[1])]
            # Return flattened list.
            return [z for s in values for z in s]

        def extract_bin_resolution(hist):
            return hist[1][1] - hist[1][0]

        def generate_num_bins(min_val, max_val, bin_resolutions):
            # Generate number of bins necessary to satisfy assumption 2
            return int(np.ceil((max_val - min_val) / bin_resolutions))

        vals = extract_vals(cur_hist) + extract_vals(prev_hist)
        bin_resolution = min(map(extract_bin_resolution, [cur_hist, prev_hist]))
        num_bins = generate_num_bins(min(vals), max(vals), bin_resolution)

        return np.histogram(vals, bins=num_bins)

    @staticmethod
    def auto_update_hist(prev_hist, cur_data):
        """
        this update_hist function directly update the previous histogram based on the current data
        it may involve padding operations
        """

        def extract_vals(hist):
            values = []
            hist_count, bin_edges = hist[0], hist[1]
            for i in range(len(hist[0])):
                values += [(bin_edges[i] + bin_edges[i + 1]) / 2] * hist_count[i]
            return values

        prev_data = extract_vals(prev_hist)
        new_data = prev_data.append(cur_data)
        return np.histogram(new_data, bins='fd')

    @staticmethod
    def _update_histogram_range(prev_hist, min_data, max_data):
        old_hist = prev_hist[0]
        bin_edges = prev_hist[1]
        bin_size = np.diff(bin_edges)[0]
        min_boundary, max_boundary = np.min(bin_edges), np.max(bin_edges)
        left_pad, right_pad = list(), list()
        while min_data < min_boundary:
            min_boundary = min_boundary - bin_size
            left_pad.append(min_boundary)
        while max_data > max_boundary:
            max_boundary = max_boundary + bin_size
            right_pad.append(max_boundary)
        new_hist = np.concatenate((np.zeros(len(left_pad)), old_hist, np.zeros(len(right_pad))))
        new_bin_edges = np.concatenate((np.array(left_pad[::-1]), bin_edges, np.array(right_pad)))
        return new_hist, new_bin_edges

    def _histogram_estimation_px(self, data, key_guess_idx):
        if self._batch_count == 0:
            if self.num_bins > 0:
                current_hist_x = np.histogram(data, bins=self.num_bins) if not self.hist_boundary else \
                    np.histogram(data, bins=self.num_bins, range=(self.hist_boundary[0], self.hist_boundary[1]))
            else:
                current_hist_x = np.histogram(data, bins='auto') if not self.hist_boundary else \
                    np.histogram(data, bins='auto', range=(self.hist_boundary[0], self.hist_boundary[1]))
            self.pdfs_for_px[key_guess_idx] = current_hist_x
        else:
            previous_hist_x = self.pdfs_for_px[key_guess_idx]
            if self.num_bins > 0:
                update_hist_x = self.update_hist(previous_hist_x, data)
            else:
                current_hist_x = np.histogram(data, bins='auto')
                update_hist_x = self.merge_hist(previous_hist_x, current_hist_x)
            self.pdfs_for_px[key_guess_idx] = update_hist_x

    def _histogram_estimation_py(self, data, time_sample_idx):
        if self._batch_count == 0:
            if self.num_bins > 0:
                current_hist_y = np.histogram(data, bins=self.num_bins) if not self.hist_boundary else \
                    np.histogram(data, bins=self.num_bins, range=(self.hist_boundary[0], self.hist_boundary[1]))
            else:
                current_hist_y = np.histogram(data, bins='auto') if not self.hist_boundary else \
                    np.histogram(data, bins='auto', range=(self.hist_boundary[0], self.hist_boundary[1]))
            self.pdfs_for_py[time_sample_idx] = current_hist_y
        else:
            previous_hist_y = self.pdfs_for_py[time_sample_idx]
            if self.num_bins > 0:
                update_hist_y = self.update_hist(previous_hist_y, data)
            else:
                current_hist_y = np.histogram(data, bins='auto')
                update_hist_y = self.merge_hist(previous_hist_y, current_hist_y)
            self.pdfs_for_py[time_sample_idx] = update_hist_y

    def _histogram_estimation_p_yx(self, key_guess_idx, c_idx, time_sample_idx, y, secret_x_i, secret_x_i_set):
        """
            1. estimate the histogram of p_yx for current batch
            2. store y_x for later processing
        """
        for secret_x_val in secret_x_i_set:
            secret_index = np.where(secret_x_i == secret_x_val)
            y_x = y[secret_index]

            if not self.pdfs_for_pyx[key_guess_idx][c_idx][time_sample_idx][secret_x_val]:
                if self.num_bins > 0:
                    current_hist_yx = np.histogram(y_x, bins=self.num_bins) if not self.hist_boundary else \
                        np.histogram(y_x, bins=self.num_bins, range=(self.hist_boundary[0], self.hist_boundary[1]))
                else:
                    current_hist_yx = np.histogram(y_x, bins='auto') if not self.hist_boundary else \
                        np.histogram(y_x, bins='auto', range=(self.hist_boundary[0], self.hist_boundary[1]))
                self.pdfs_for_pyx[key_guess_idx][c_idx][time_sample_idx][secret_x_val] = current_hist_yx
            else:
                previous_hist_yx = self.pdfs_for_pyx[key_guess_idx][c_idx][time_sample_idx][secret_x_val]
                if self.num_bins > 0:
                    update_hist_yx = self.update_hist(previous_hist_yx, y_x)
                else:
                    current_hist_y = np.histogram(y_x, bins='auto')
                    update_hist_yx = self.merge_hist(previous_hist_yx, current_hist_y)
                self.pdfs_for_pyx[key_guess_idx][c_idx][time_sample_idx][secret_x_val] = update_hist_yx

    def _store_yx(self, key_guess_idx, c_idx, time_sample_idx, y, secret_x_i, secret_x_i_set):
        for secret_x_val in secret_x_i_set:
            secret_index = np.where(secret_x_i == secret_x_val)
            y_x = y[secret_index]
            # if the y_x is firstly stored
            if not isinstance(self.y_x[key_guess_idx][c_idx][time_sample_idx][secret_x_val], np.ndarray):
                self.y_x[key_guess_idx][c_idx][time_sample_idx][secret_x_val] = y_x
            else:
                prev = self.y_x[key_guess_idx][c_idx][time_sample_idx][secret_x_val]
                self.y_x[key_guess_idx][c_idx][time_sample_idx][secret_x_val] = np.concatenate((prev, y_x), axis=0)

    @staticmethod
    def _cal_integration_term(p_y, p_yx, yx):
        p_y_dist = rv_histogram(p_y)
        p_yx_dist = rv_histogram(p_yx)
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
        p_x_dist = rv_histogram(p_x)
        # calculate eqn (4)
        for xi in range(len_x):
            # integration part
            yx = np.unique(yx_total[xi])
            if not p_yx[xi]:
                continue
            integrate_term = self._cal_integration_term(p_y, p_yx[xi], yx)
            integrate_res = integrate.trapezoid(integrate_term, yx)
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
        # store the total secret x
        self.secret_x = secret_x if not isinstance(self.secret_x, np.ndarray) else np.concatenate(
            (self.secret_x, secret_x), axis=0)

        # store the total y
        batch_leakages = batch.leakages
        self.y_total = batch_leakages if not isinstance(self.y_total, np.ndarray) else np.concatenate(
            (self.y_total, batch_leakages), axis=0)
        # estimate the histogram of p_y for current batch
        for i in range(self.number_of_time_samples):
            y = np.array(batch_leakages[:, i], ndmin=2).T
            self._histogram_estimation_py(y, i)

        print('[INFO] Processing pdfs estimation...')
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
                p_yx = self.pdfs_for_pyx[i][0][j]  # list
                yx = self.y_x[i][0][j]  # list
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
        return self._mutual_information, self._p_value

    def _clean(self):
        del self._mutual_information
        del self._p_value
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
