from math import inf

import numpy as np
from scipy import integrate
from scipy.stats import mvn, multivariate_normal, binom
from Lib_SCA.lascar import numerical_success_rate
from sklearn.neighbors import KernelDensity
from pyitlib import discrete_random_variable as drv

from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
import matplotlib.pyplot as plt

container = SimulatedPowerTraceContainer('normal_simulated_traces.yaml')
# container = SimulatedPowerTraceFixedRandomContainer('fixed_random_traces.yaml')
yyy = container[:3000].leakages


#
# container.plot_traces(list(range(container.number_of_traces)))
# container.plot_traces(list(range(1, container.number_of_traces, 2)))
# print(container[25].value['trace_idx'])
# snr_theo = container.calc_snr('theo')

# snr_real = container.calc_snr('real')
#
# plt.figure(1)
# plt.title('snr')
# plt.plot(snr_theo)
# plt.plot(snr_real)
# plt.legend(['theoretical snr', 'actual snr'])
# plt.show()

# mean = np.array([0.5, 0.5, 0.3]).T
# cov = np.array([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.25, 0.3, 1]]).T
# dis = multivariate_normal(mean=mean, cov=cov)
# y = dis.cdf(np.array([0, 0, 0]))
# # x = np.outer(a[:, 0], a[:, 0].T) + np.outer(a[:, 1], a[:, 1].T)
# print(y)

# x = np.array([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.25, 0.3, 1]])
# y = numerical_success_rate(x, 0, 5).eval()

# x = np.random.binomial(n=8, p=0.5, size=(1000, 1))
# x = np.random.normal(0, 1.0, (1000, 1))
# y = np.sum(x, axis=0)
# print(x[0, 9])
# occ, bin_edges = np.histogram(x, bins='auto')
# bin_values = np.digitize(x, bin_edges)
# o = np.array(bin_values, dtype=int)
# h = np.array(x, dtype=int)
# xxx = drv.information_mutual(o, h)
# bd = np.sum(x, axis=0)
# bdd = np.array(bd, ndmin=2)
# kde = KernelDensity(kernel='epanechnikov', bandwidth=0.5).fit(x)
# y = np.linspace(0, np.max(x), 1000)
# y = np.array(y, ndmin=2).T
# z = kde.score_samples(y)
# plt.figure(1)
# plt.plot(y, z)
# plt.show()
# y = kde.score_samples(np.ones((1, 1)) * 1.5)
# z = kde.score_samples(x)


# def f(x, kde):
#     x = np.ones((1, 1)) * x
#     return np.exp(kde.score_samples(x))
def f(x):
    return x ** 2
#
#
x = np.linspace(0, 100, 4)
y = f(x)
print(integrate.simpson(y, x))
# print(integrate.simpson(y, dx=0.5))
print(np.sum(y))
print(100 ** 3 / 3)
# print(integrate.trapezoid(y, x))

# v = integrate.quad(f, 1, 50, args=kde)
# y = kde.score_samples(x)
# y1 = np.exp(y)
# plt.figure(1)
# plt.plot(x, y1)
# plt.show()

# aa = np.ones((3, 2))
# print(aa[:, 1].shape)
#
# a = [1, 2, 3, 2, 2, 1, 4, 3]
# b = [1, 2, 3, 4, 5, 6, 7, 8]
# a1 = np.array(a)
# b = np.array(b, ndmin=2).T
# ua1 = np.unique(a1, axis=0)
# d = np.where(a1 == ua1[0])
# c = b[d]
# p_x = binom.pmf(5, 8, 0.5)
# print(p_x)

# arr = np.arange(27).reshape((3, 3, 3))
# b = arr[:, 2, :] > arr[:, 0, :]
# arr[1:, :] = b
# c = np.sum(arr[1:, :], axis=1)
# a = np.arange(20)
# x = np.random.choice(a, 20, replace=False)
# y = np.random.choice(a, 20, replace=False)
# z = np.random.choice(a, 20, replace=False)