import numpy as np
from scipy import integrate
from sklearn.neighbors import KernelDensity

from .engine import Engine


class MiEngine(Engine):
    """
    MiEngine is an specialized Engine which is used to calculate the mutual information of the leakage
    the implementation refers to the paper related to the continuous mutual information (CMI) proposed in
    Chothia, Tom, and Apratim Guha. "A statistical test for information leaks using continuous mutual information."
    2011 IEEE 24th Computer Security Foundations Symposium. IEEE, 2011.
    """

    def __init__(self, name, kernel='epanechnikov', jit=False):
        """
        MiEngine
        :param name: name of the engine
        :param kernel: kernel type used in the pdf estimation
        """
        self.kernel = kernel
        self.jit = jit
        Engine.__init__(self, name)

    def _initialize(self):
        self._mutual_information = np.zeros(self._session.leakage_shape, dtype=np.double, )
        self._batch_count = np.zeros(self._session.leakage_shape, dtype=np.int32, )

        self.size_in_memory += self._mutual_information.nbytes

        if self.jit:
            from numba import jit

            @jit(nopython=True)
            def jitted_update(batchvalues, batchleakages, pfunc=self._partition_function, psize=self._partition_size,
                              rng2idx=self._partition_range_to_index, order=self._order):
                pcount = np.zeros((psize,), dtype=np.uint32)
                acc_x = np.zeros((order, psize, batchleakages.shape[1]), dtype=np.double)

                for pv in np.arange(batchvalues.shape[0]):
                    idx = rng2idx[pfunc(batchvalues[pv])]
                    pcount[idx] += 1
                    acc_x[0, idx] += batchleakages[pv]
                    for o in range(1, order):
                        acc_x[o, idx] += np.power(batchleakages[pv], o + 1)

                return pcount, acc_x

            def new_update(batch):
                pcount, acc_x = jitted_update(batch.values, batch.leakages)
                self._partition_count += pcount
                self._acc_x_by_partition += acc_x

            self._update = new_update
        else:
            self._update = self._base_update

    def _base_update(self, batch):
        """
        1. we use kernel estimation to estimate the pdf of p(y|x), kernel used in CMI is epanechnikov kernel and bandwidth
        can be set with general purpose (5) in the paper
        2. we assume the distribution of input secret x (plaintext) is known as an uniform distribution has a pmf with
        1 / (b-a+1), although we have more than one plaintext in one trace, as pmf of uniform distribution is always a
        constant, we see them in one uniform distribution with range [0, 255], so the p(x) = 1/256 (constant)
        3. we use the equation 4 in the paper to calculate continuous mutual information
        """
        batch_leakages = batch.leakages
        batch_size = batch_leakages.shape[0]
        time_samples = batch_leakages.shape[1]
        p_x = 1.0 / 256
        for i in range(time_samples):
            bandwidth = 1.06 * np.sqrt(np.var(batch_leakages[:, i])) * batch_size ** (-1 / 5)
            lk = np.array(batch_leakages[:, i], ndmin=2).T
            kde = KernelDensity(kernel=self.kernel, bandwidth=bandwidth).fit(lk)
            log_p_yx = kde.score_samples(lk)
            p_yx = np.exp(log_p_yx)
            sum_term = np.sum(p_yx, axis=0) * p_x
            lower_bound = min(batch_leakages[:, i])
            upper_bound = max(batch_leakages[:, i])

            # integration part
            # function build
            def f(x, pdf, s_term):
                x = np.ones((1, 1)) * x
                pyx = np.exp(pdf.score_samples(x))
                if not pyx:
                    return 0.0
                else:
                    return pyx * np.log(pyx / s_term)

            integrate_term = integrate.quad(f, lower_bound, upper_bound, args=(kde, sum_term))[0]
            self._mutual_information[i] += batch_size * p_x * integrate_term
            self._batch_count[i] += 1

    def _finalize(self):
        return self._mutual_information / self._batch_count

    def _clean(self):
        del self._mutual_information
        del self._batch_count
        self.size_in_memory = 0


class MiTestEngine(Engine):
    """
    PartitionEngine is an abstract specialized Engine which role is to partition the 'leakages', according
    to a 'partition_value' computed from the 'values'.

    The partition_value is computed by a function called 'partition', which outputs are in partition_range.

    partition_value = partition(value)
    0 <= partition_value < partition_size
    """

    def __init__(self, name, partition_function, partition_range, order, jit=True):
        """
        PartitionEngine
        :param name: the name chosen for the Engine
        :param session: the Session that will drive it
        :param partition_function: a function (or callable) which will be applied to the trace values and return a positive integer
        :param partition_range: the possible partition_values
        :param order: the order needed by the engine ( order=1: sum of leakages, order=2: sum of square of leakages,...)

        """

        if isinstance(partition_range, int):
            self._partition_range = range(partition_range)
        else:
            self._partition_range = partition_range

        self._partition_range = np.array(self._partition_range, dtype=np.uint32)
        self._partition_size = len(self._partition_range)
        self._partition_range_to_index = np.zeros((self._partition_range.max() + 1,), dtype=np.uint32)
        for i, j in enumerate(self._partition_range):
            self._partition_range_to_index[j] = i

        self._order = order

        self.jit = jit
        if jit:
            try:
                from numba import jit
            except Exception:
                raise Exception(
                    "Cannot jit without Numba. Please install Numba or consider turning off the jit option"
                )
            self._partition_function = jit(nopython=True)(partition_function)
        else:
            self._partition_function = partition_function
        Engine.__init__(self, name)

    def _initialize(self):

        self._acc_x_by_partition = np.zeros(
            (self._order, self._partition_size) + self._session.leakage_shape,
            dtype=np.double,
        )

        # acc_x_by_partition[i,j,k] = sum( (leakages[k])**i | partition = j)

        self._partition_count = np.zeros((self._partition_size,), dtype=np.double)

        self.size_in_memory += self._acc_x_by_partition.nbytes
        self.size_in_memory += self._partition_count.nbytes
        if self.jit:
            from numba import jit

            @jit(nopython=True)
            def jitted_update(batchvalues, batchleakages, pfunc=self._partition_function, psize=self._partition_size,
                              rng2idx=self._partition_range_to_index, order=self._order):
                pcount = np.zeros((psize,), dtype=np.uint32)
                acc_x = np.zeros((order, psize, batchleakages.shape[1]), dtype=np.double)

                for pv in np.arange(batchvalues.shape[0]):
                    idx = rng2idx[pfunc(batchvalues[pv])]
                    pcount[idx] += 1
                    acc_x[0, idx] += batchleakages[pv]
                    for o in range(1, order):
                        acc_x[o, idx] += np.power(batchleakages[pv], o + 1)

                return pcount, acc_x

            def new_update(batch):
                pcount, acc_x = jitted_update(batch.values, batch.leakages)
                self._partition_count += pcount
                self._acc_x_by_partition += acc_x

            self._update = new_update
        else:
            self._update = self._base_update

    def _base_update(self, batch):
        partition_values = list(map(self._partition_function, batch.values))
        for i, v in enumerate(partition_values):
            idx = self._partition_range_to_index[v]
            self._partition_count[idx] += 1
            for o in range(0, self._order):
                self._acc_x_by_partition[
                    o, idx
                ] += np.power(batch.leakages[i], o + 1, dtype=np.double)

    def _finalize(self):
        pass

    def _clean(self):
        del self._acc_x_by_partition
        del self._partition_count
        self.size_in_memory = 0

    def get_mean_by_partition(self):
        """
        Compute a np.array containing the means by partition

        E[ leakage | partition(value) = i) for i in partition_values

        :return: np.array containing the means by partition.
        """

        acc = np.zeros(self._acc_x_by_partition.shape[1:], np.double)
        for v in self._partition_range:
            i = self._partition_range_to_index[v]
            acc[i] = self._acc_x_by_partition[0, i] / self._partition_count[i]

        return acc
