import numpy as np
from scipy import integrate
from sklearn.neighbors import KernelDensity
from scipy.stats import binom
from tqdm import tqdm

from . import GuessEngine


class CMI_Engine(GuessEngine):
    """
    CMI_Engine is an specialized Engine which is used to calculate the mutual information of the leakage as well as the
    statistical test
    the implementation refers to the paper related to the continuous mutual information (CMI) proposed in
    Chothia, Tom, and Apratim Guha. "A statistical test for information leaks using continuous mutual information."
    2011 IEEE 24th Computer Security Foundations Symposium. IEEE, 2011.
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
        if self.contain_test:
            self._mutual_information = np.zeros((self._number_of_guesses,
                                                 self.num_shuffles + 1) + self._session.leakage_shape, np.double)
            self._batch_count = np.zeros((self._number_of_guesses,
                                          self.num_shuffles + 1) + self._session.leakage_shape, dtype=np.int32, )
        else:
            self._mutual_information = np.zeros((self._number_of_guesses, ) + self._session.leakage_shape, np.double)
            self._batch_count = np.zeros((self._number_of_guesses, ) + self._session.leakage_shape, dtype=np.int32, )

        self.size_in_memory += self._mutual_information.nbytes
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
        # p_x (binomial distribution)
        p_x = binom.pmf(secret_x, n=8, p=0.5)
        for i in tqdm(range(self._number_of_guesses)):
            print('[INFO] Processing Key Guess', i)
            p_x_set = np.unique(p_x[:, i])  # no repeated item
            for j in tqdm(range(time_samples)):
                lk_total = np.array(batch_leakages[:, j], ndmin=2).T
                kde_set, cmi = self._cal_mutual_information(p_x[:, i], p_x_set, lk_total, batch_size)
                if self.contain_test:
                    # real mutual information
                    self._mutual_information[i][0][j] += cmi
                    self._batch_count[i][0][j] += 1

                    # statistical test
                    for l in range(1, self.num_shuffles + 1):
                        np.random.shuffle(lk_total)
                        cmi_t = self._cal_mutual_information_test(p_x[:, i], p_x_set, lk_total, kde_set)
                        self._mutual_information[i][l][j] += cmi_t
                        self._batch_count[i][l][j] += 1
                else:
                    # real mutual information
                    self._mutual_information[i][j] += cmi
                    self._batch_count[i][j] += 1

    @staticmethod
    def _cal_integration_term(lk, kde_set, p_x_set, set_id):
        lk = np.array(lk, ndmin=2).T
        cur_pdf = kde_set[set_id]
        set_length = len(kde_set)
        pyx = np.exp(cur_pdf.score_samples(lk))
        sum_term = 0
        for si in range(set_length):
            sum_term += p_x_set[si] * np.exp(kde_set[si].score_samples(lk))
        # extreme value processing
        sum_term_mask = sum_term == 0.0
        sum_term[sum_term_mask] = 1.0
        log_term = pyx / sum_term
        log_term_mask = log_term == 0.0
        log_term[log_term_mask] = 1.0
        return pyx * np.log(log_term)

    def _cal_mutual_information(self, p_x, p_x_set, lk, batch_size):
        kde_set = []  # for each element in set, estimate corresponding pdf of y
        lk_set = []
        # calculate related terms
        for k1 in range(p_x_set.shape[0]):
            index = np.where(p_x == p_x_set[k1])
            lk_set_i = lk[index]
            bandwidth = 1.06 * np.sqrt(np.var(lk)) * batch_size ** (-1 / 5)
            kde = KernelDensity(kernel=self.kernel, bandwidth=bandwidth).fit(lk_set_i)
            kde_set.append(kde)
            lk_set.append(lk_set_i)
        cmi = 0
        # calculate eqn (4)
        for k2 in range(p_x_set.shape[0]):
            # integration part
            lk_series = np.unique(lk_set[k2])
            f_lk = self._cal_integration_term(lk_series, kde_set, p_x_set, k2)
            integrate_term = integrate.trapezoid(f_lk, lk_series)
            cmi += p_x_set[k2] * integrate_term
        return kde_set, cmi

    def _cal_mutual_information_test(self, p_x, p_x_set, lk, kde_set):
        lk_set = []
        # calculate related terms
        for k1 in range(p_x_set.shape[0]):
            index = np.where(p_x == p_x_set[k1])
            lk_set_i = lk[index]
            lk_set.append(lk_set_i)
        cmi = 0
        # calculate eqn (4)
        for k2 in range(p_x_set.shape[0]):
            # integration part
            lk_series = np.unique(lk_set[k2])
            f_lk = self._cal_integration_term(lk_series, kde_set, p_x_set, k2)
            integrate_term = integrate.trapezoid(f_lk, lk_series)
            cmi += p_x_set[k2] * integrate_term
        return cmi

    def _finalize(self):
        if self.contain_test:
            mean_mi = self._mutual_information / self._batch_count
            real_mi = mean_mi[:, 0, :]
            for j in range(1, self.num_shuffles+1):
                is_pass = mean_mi[:, j, :] < mean_mi[:, 0, :]
                mean_mi[:, j, :] = is_pass
            test_score = np.sum(mean_mi[:, 1:, :], axis=1)
            return real_mi, test_score
        else:
            return self._mutual_information / self._batch_count

    def _clean(self):
        del self._mutual_information
        del self._batch_count
        self.size_in_memory = 0

# class CMI_Test_Engine(Engine):
#     """
#     MiTestEngine is an specialized Engine which is used to calculate the mutual information of
#     the leakage as well as the statistic test related to it
#     the implementation refers to the paper related to the continuous mutual information (CMI) proposed in
#     Chothia, Tom, and Apratim Guha. "A statistical test for information leaks using continuous mutual information."
#     2011 IEEE 24th Computer Security Foundations Symposium. IEEE, 2011.
#     """
#
#     def __init__(self, name, kernel='epanechnikov', k=10, jit=False):
#         """
#         MiEngine
#         :param name: name of the engine
#         :param kernel: kernel type used in the pdf estimation
#         """
#         self.kernel = kernel
#         self.k = k
#         self.jit = jit
#         Engine.__init__(self, name)
#
#     def _initialize(self):
#         """
#         1. we use kernel estimation to estimate the pdf of p(y|x), kernel used in CMI is epanechnikov kernel and bandwidth
#         can be set with general purpose (5) in the paper
#         2. we assume the distribution of input secret x (plaintext) is known as an uniform distribution has a pmf with
#         1 / (b-a+1), although we have more than one plaintext in one trace, as pmf of uniform distribution is always a
#         constant, we see them in one uniform distribution with range [0, 255], so the p(x) = 1/256 (constant)
#         3. we use the equation 4 in the paper to calculate continuous mutual information
#         """
#         self._mutual_information = np.zeros((self.k + 1,) + self._session.leakage_shape, dtype=np.double, )
#         self._batch_count = np.zeros(self._session.leakage_shape, dtype=np.int32, )
#
#         self.size_in_memory += self._mutual_information.nbytes
#
#         if self.jit:
#             self._update = self._base_update
#         else:
#             self._update = self._base_update
#
#     def _base_update(self, batch):
#         batch_leakages = batch.leakages
#         batch_size = batch_leakages.shape[0]
#         time_samples = batch_leakages.shape[1]
#         p_x = 1.0 / 256
#         from Lib_SCA.lascar import SimulatedPowerTraceContainer
#         reference_container = [SimulatedPowerTraceContainer(batch_size)[:batch_size]] * self.k
#         for i in range(time_samples):
#             bandwidth = 1.06 * np.sqrt(np.var(batch_leakages[:, i])) * batch_size ** (-1 / 5)
#             lk = np.array(batch_leakages[:, i], ndmin=2).T
#             kde = KernelDensity(kernel=self.kernel, bandwidth=bandwidth).fit(lk)
#             lk_test = np.unique(lk, axis=0)
#             log_p_yx = kde.score_samples(lk_test)
#             p_yx = np.exp(log_p_yx)
#             sum_term = np.sum(p_yx, axis=0) * p_x
#
#             lower_bound = np.min(lk)
#             upper_bound = np.max(lk)
#
#             # integration part
#             # function build
#             def f(x, pdf, s_term):
#                 x = np.ones((1, 1)) * x
#                 pyx = np.exp(pdf.score_samples(x))
#                 if not pyx:
#                     return 0.0
#                 else:
#                     return pyx * np.log(pyx / s_term)
#
#             integrate_term = integrate.quad(f, lower_bound, upper_bound, args=(kde, sum_term))[0]
#             self._mutual_information[0][i] += batch_size * p_x * integrate_term
#             self._batch_count[i] += 1
#             # statistical test
#             for j in range(self.k):
#                 ref_lk = np.array(reference_container[j].leakages[:, i], ndmin=2).T
#                 ref_lk_test = np.unique(ref_lk, axis=0)
#                 ref_log_p_yx = kde.score_samples(ref_lk_test)
#                 ref_p_yx = np.exp(ref_log_p_yx)
#                 ref_sum_term = np.sum(ref_p_yx, axis=0) * p_x
#
#                 ref_lower_bound = np.min(ref_lk)
#                 ref_upper_bound = np.max(ref_lk)
#
#                 # integration part
#
#                 integrate_term = integrate.quad(f, ref_lower_bound, ref_upper_bound, args=(kde, ref_sum_term))[0]
#                 self._mutual_information[j + 1][i] += batch_size * p_x * integrate_term
#
#     def _finalize(self):
#         mean_mi = self._mutual_information / self._batch_count
#         for i in range(1, mean_mi.shape[0]):
#             for j in range(mean_mi.shape[1]):
#                 if mean_mi[i][j] < mean_mi[0][j]:
#                     mean_mi[i][j] = 1
#                 else:
#                     mean_mi[i][j] = 0
#         mutual_information = mean_mi[0, :]
#         test_score = np.sum(mean_mi[1:, :], axis=0)
#         return mutual_information, test_score
#
#     def _clean(self):
#         del self._mutual_information
#         del self._batch_count
#         self.size_in_memory = 0
