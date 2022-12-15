from math import inf

import numpy as np
import sklearn.feature_selection as fs
from PyAstronomy import pyaC
from scipy import integrate
from scipy.stats import mvn, multivariate_normal, binom, norm, rv_histogram, rv_discrete, bernoulli
from sklearn.cluster import KMeans

from Lib_SCA.lascar import numerical_success_rate
from sklearn.neighbors import KernelDensity
from pyitlib import discrete_random_variable as drv

from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
import matplotlib.pyplot as plt

# container = SimulatedPowerTraceContainer('normal_simulated_traces.yaml')
# container = SimulatedPowerTraceFixedRandomContainer('fixed_random_traces.yaml')
# yyy = container[:3000].leakages

#
# container.plot_traces(list(range(container.number_of_traces)))
# container.plot_traces(list(range(1, container.number_of_traces, 2)))
# print(container[25].value['trace_idx'])
# snr_theo = container.calc_snr('theo')
# a = [i for i in range(0)]
# a = np.arange(10)
# b = a[6:(3-1)*2+6+1:2]
# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# c = np.ones((5, 1))
# x = np.array([[0, 0, 0],
#             [1, 1, 0],
#             [2, 0, 1],
#             [2, 0, 1],
#             [2, 0, 1]])
# y = np.array([0, 1, 2, 2, 1])
# xx = fs.mutual_info_classif(c, x[:, 2])

# x = np.array([0, 1, 0, 1, 1, 1, 1, 0, 1])
# z = np.argmax(x)
# # zz = np.where(x == 0)
# y = np.array([1, 0, 0, 0, 0, 1, 0, 1, 1])
#
# b = np.array([[1, 5, 9], [2, 4, 6]])
# bb = np.linalg.norm(b, ord=inf, axis=0)
# ccc = b[:, [0, 1]]
# c = np.array(np.max(b, axis=1), ndmin=2).T
# d = np.concatenate((c, c), axis=1)
#
# dic = dict()
# dic.update({'name': 123})

# Generate some 'data'
x = np.arange(2.)**2
y = np.sin(x)

# Get coordinates and indices of zero crossings
# xc, xi = pyaC.zerocross1d(x, y, getIndices=True)
# fig, axs = [None] * 2, [None] * 2
# fig[0], axs[0] = plt.subplots(1, 2, figsize=(16, 6))
# fig[1], axs[1] = plt.subplots(1, 2, figsize=(16, 6))
xx = np.array((1))
# a = np.random.randint(0, 5, (2, 2, 2))
# b = np.random.randint(0, 5, (5, 5))
# print(b[1:2, 4])
# print(np.mean(a, axis=0))
# print(np.var(a, axis=0))
# b = a.flatten()
# u, s, et = np.linalg.svd(a)
# b = np.array(et[:2], ndmin=2)

# b = np.random.rand(3, 3)
# b[0][0] = 1
# b[1][1] = 1
# b[0][2] = 1
# b[2][2] = 1
# r = bernoulli.rvs(b, size=(3, 3))
# c = np.arange(1, 100, 0.5)
# print(np.diag_indices_from(a))
# print(a)
# a[a > 5] = -1
# print(a)
# b = np.max(a)
# b = np.ones((3, 2))
# print(a ** 2)
# print(np.linalg.norm(a, axis=(0, 1)))
# snr_real = container.calc_snr('real')

# plt.figure(1)
# plt.title('snr')
# plt.plot(snr_theo)
# plt.plot(snr_real)
# plt.legend(['theoretical snr', 'actual snr'])
# plt.show()
# x = np.random.normal(0, 1.0, (1000, 1))
# y = np.random.normal(0, 1.0, (1000, ))
#
# a = [[[[None] * 10
#        for _ in range(10)]
#       for __ in range(10)]
#      for ___ in range(10)]
# x = np.random.normal(0, 1.0, (1000, 3))
# c = np.random.randn(10, 18)
# y = x.tolist()
# # yy = [np.array(x[:, i]) for i in range(256)]
# x = [[1] * 1000 for _ in range(1000)]
# cov = np.array([[1, 2, 2, 3, 5], [1, 1, 1, 2, 5], [1, 2, 1, 4, 5], [1, 4, 4, 2, 5]])
# s = cov.tolist()
# x = np.array([1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.5, 0.5, 2.1, 2.56, 2.5, 5], ndmin=2).T
# print(isinstance(x, np.ndarray))
# print(np.unique(x))data
from collections import Counter

# xcx = np.array([0, 1, 3])
# covvv = cov[:, xcx]
# y = np.zeros((1000000, 10000))
# y = np.array([-5, -4, -4, -3, -0.5, 0, 0.25, 3, 4, 6], ndmin=2).T
# # a = Counter(x.flatten())
# hist = np.histogramdd(c, bins=2)
# hist_count = hist[0]
#
# # pdff = rv_histogram(hist)
# x = np.sum(hist_count, axis=(1, 2, 3))
# a = np.vstack((x, np.sum(hist_count, axis=(0, 2, 3))))
# b = np.vstack((a, np.sum(hist_count, axis=(0, 1, 3))))
# c = np.vstack((b, np.sum(hist_count, axis=(0, 1, 2))))
#
# cl = c.tolist()
# c1 = np.array(hist[1])

# hist_dist = rv_histogram((cl, hist[1]))

# b = np.array([0.15, 0.8, 0.19])
# print(np.concatenate((b, hist[0]))) s
# print(min(a.keys()))
# print(type((x, a)))
# print(hist_dist.pdf([0.1, 0.3, 0.5]))
from bisect import bisect_left

# print(bisect_left(np.array([-7.25000,-4.80000,-2.35000,0.10000,2.55000,5.00000,7.45000]), 5))


# def update_hist(prev_hist, cur_data):
#     """
#         this update_hist function directly update the previous histogram based on the current data
#         it may involve padding operations
#         """
#     old_hist = prev_hist[0]
#     bin_edges = prev_hist[1]
#     bin_size = np.diff(bin_edges)[0]
#     min_boundary, max_boundary = np.min(bin_edges), np.max(bin_edges)
#     counter_dic = Counter(cur_data.flatten())
#     min_data, max_data = min(counter_dic.keys()), max(counter_dic.keys())
#     left_pad, right_pad = list(), list()
#     while min_data < min_boundary:
#         min_boundary = min_boundary - bin_size
#         left_pad.append(min_boundary)
#     while max_data > max_boundary:
#         max_boundary = max_boundary + bin_size
#         right_pad.append(max_boundary)
#     new_hist = np.concatenate((np.zeros(len(left_pad)), old_hist, np.zeros(len(right_pad))))
#     new_bin_edges = np.concatenate((np.array(left_pad[::-1]), bin_edges, np.array(right_pad)))
#     for data_i in cur_data:
#         index = bisect_left(new_bin_edges, data_i) - 1
#         new_hist[index] += 1
#     res = (new_hist, new_bin_edges)
#     return res

# xx = update_hist(hist, y)

# print(bisect_left(hist[1], b))
# z = sklearn.feature_selection.mutual_info_regression(x, y)
# mean = np.array([0.5, 0.5, 0.3]).T
# cov = np.array([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.25, 0.3, 1]]).T
# dis = multivariate_normal(mean=mean, cov=cov)
# y = dis.cdf(np.array([0, 0, 0]))
# # x = np.outer(a[:, 0], a[:, 0].T) + np.outer(a[:, 1], a[:, 1].T)
# print(y)

# x = np.array([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.25, 0.3, 1]])
# y = numerical_success_rate(x, 0, 5).eval()

# x = np.random.binomial(n=8, p=0.5, size=(1000, 1))
# p_x = binom.pmf([0, 1, 2, 3, 4, 5, 6, 7, 8], n=8, p=0.5)
# x = np.random.normal(0, 1.0, (1000, 1))
# y = np.sum(x, axis=0)
# print(x[0, 9])
# x = np.zeros(100)
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

# z = norm.cdf(1, loc=1, scale=0.16)
# def f(x, kde):
#     x = np.ones((1, 1)) * x
#     return np.exp(kde.score_samples(x))
# def f(x):
#     return x ** 2
# #
# #
# x = np.linspace(0, 100, 4)
# y = f(x)
# print(integrate.simpson(y, x))
# # print(integrate.simpson(y, dx=0.5))
# print(np.sum(y))
# print(100 ** 3 / 3)
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
# a = np.array(a)
# a1 = np.copy(a)
# np.random.shuffle(a)
# print(a)
# print(a1)
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
