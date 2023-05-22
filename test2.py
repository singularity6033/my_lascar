import collections
import os
from math import inf

import pandas as pd
from numpy import unravel_index
from scipy.stats import norm, bernoulli, pearsonr, moment, describe, ks_2samp, binom
from sklearn.preprocessing import OneHotEncoder
# x = binom.pmf([0, 1, 2, 3, 4, 5, 6, 7, 8], 8, 0.5)

from Lib_SCA.hdf5_files_import import read_hdf5_proj
from Lib_SCA.lascar import SimulatedPowerTraceFixedRandomContainer
import numpy as np
from scipy.linalg import eigvals, eigvalsh
from math import factorial
from Lib_SCA.lascar import hamming
from Lib_SCA.lascar.tools.aes import sbox

# a = np.array([[1, 2], [4, 5]])
# b = np.linalg.norm(a, ord=inf, axis=0, keepdims=True)
# from configs.simulation_configs import fixed_random_traces
#
# container = SimulatedPowerTraceFixedRandomContainer(config_params=fixed_random_traces)
# xx = container[0].value
# dtype = np.dtype([('trace_idx', np.uint8, ())])
# value = np.zeros((), dtype=dtype)
#
# for i in range(100):
#     value = np.zeros((), dtype=dtype)
#     value['trace_idx'] = i
#     if i == 0:
#         values = value
#     else:
#         values = np.hstack([values, value])
# xxx = values
from bisect import bisect_left, bisect_right, bisect

# bb = bernoulli.rvs(0.5, size=(factorial(1262), 1))
# levels = np.arange(0, 10 + (10 - 0) /10, (10 - 0) /10).tolist()
# l = [i for i in range(11)]
# x = np.random.randint(0, 10, (6, 8))
# mx = np.max(x, axis=(1, 2))
# x = np.zeros(10)
# x[3] = 2
# x[4] = 2
# y = x[x == 2].flatten().tolist()
# print([1, 2, 3, 4] + [89])
# new_hist = np.concatenate((np.ones(len([1])), x, np.zeros(len([]))))
# print(0 % 1)
# x = np.searchsorted(l, [0, 0.5, 2, 10, 3.5], side='left')
# index = bisect_left(l, [0.5, 2, 10, 3.5])
# a = np.random.randn(3, 6)
# # tmp = a
# # aa = a.flatten()
# # # b = a[:, 1]
# # c = np.repeat(aa, 3)
# # d = np.reshape(c, (-1, 3*6))
# # for i in range(3):
# #     tmp = np.concatenate((tmp, a), axis=1)
# # s = 'sd_l2'
# # x = s[:2]
# b = np.where(a > 0)
# a[b[0], b[1]] = 100
# old_hist_count = np.reshape(a, (3, 3, -1))
# left_padding = np.zeros((3, 3, 1))
# right_padding = np.zeros((3, 3, 1))
# updated_hist_count = np.concatenate((left_padding, old_hist_count, right_padding), axis=-1)
# updated_hist_count_1 = np.pad(old_hist_count,
#                             ((0, 0), (0, 0), (1, 1)),
#                             'constant',
#                             constant_values=(0, 0))
# hist_counts = np.reshape(updated_hist_count, (3, 12))
# x = not a.all()
# y = np.empty(10)
# print(y.shape[0])
# z = not y.all()
# idx = np.triu_indices(5, 1)
# b = a[idx]
# 5)
# def _calc_sup_norm(a):
#     size = a.shape[0]
#     # try to speedup the calculation
#     tmp1 = np.repeat(a, size, axis=0)
#     tmp2 = np.tile(a, (size, 1))
#     diff = np.abs(tmp1 - tmp2)
#     sup_norm = np.max(diff, axis=1)
#     adj_matrix = np.reshape(sup_norm, (size, size), order='C')
#     upper_tri_idx = np.triu_indices(size)
#     xx = adj_matrix[upper_tri_idx]
#     return sup_norm
# a = np.array([100, -2, 3])
# c = np.array([[1, 0.5, 0.6], [0.5, 1, 0.8], [2, 2, 3]])
# v = c == 0.5
# tmp1 = np.repeat(c, 3, axis=1)
# tmp2 = np.tile(c, (1, 3))
# # d = np.searchsorted([1, 2, 3, 4, 5], np.array([[1, 2.5], [3, 5]]), side='left')
# # _calc_sup_norm(a)
# k = bernoulli.rvs(0.9)
# d = np.delete(c, 0, 0)
# d = np.delete(d, 0, 1)
# x = list(range(10))
# x = np.sort(a)
# xx = np.argsort(a)
# print(True * False)
# # y = np.argwhere(a < 0)
# v = np.array([list(range(10)) for _ in range(10)])
# v_tmp = np.ndarray.flatten(v)
# v_tmp = np.delete(v_tmp, range(0, len(v_tmp), len(v) + 1), 0)
# v_set = v_tmp.reshape(len(v), len(v) - 1)
# tmp1 = np.repeat(a.T, 2, axis=1)
# tmp2 = np.tile(a, (2, 1))
# d = np.sum(c, axis=1, keepdims=True)
# print(np.dot(d, d.T))
# d = np.diag(c)
# data = np.array([0, 0, 1, 2, 0, 1, 1, 1, 1, 1])
# xx = np.repeat(data, 3, axis=1)
# res = ks_2samp(data, data)
# print(res.statistic)
# print(len(data))
# xx = data.reshape((5, 2))
# # enc = OneHotEncoder()
# # y = enc.fit_transform(data).toarray()
# u, s, et = np.linalg.svd(c)
# x = np.array(np.sum(c, 0), ndmin=2)
# y = c / x
# uu = np.sum(np.sqrt(u ** 2), axis=0)
# print(np.dot(u[:, :2], u[:, :2].T))
# d = np.diag(np.sum(c, axis=1))
# print(np.count_nonzero(a == 2))
# print(list(range(0.0001, 5.22, 0.02)))
# import numpy as np
# from TracyWidom import TracyWidom
#
# x = np.arange(*(-389, 360), dtype=np.int32).astype(float) * 1.e-2
# xx = x <= -233
# # x = np.linspace(-10, 10, 101)
# # y = np.linspace(0, 1, 101)
# tw = TracyWidom(1)
# # # pdf = tw.pdf(x)
# cdf = tw.cdf(max(1, np.sum(x <= -233)))
# cdfinv = tw.cdfinv(y)

# from scipy.io import loadmat
# annots = loadmat('./Lib_SCA/lascar/engine/TW_beta1_CDF.mat')

# b = np.repeat(a, 3, axis=1)
# z = np.array([100, 100], ndmin=2)
# b = np.array([[7, 8, 9], [0, -1, -3]])
# x = [7, 8, 9] + [0, -3, 100]
# # y = np.array(x)
# z = np.reshape(x, (-1, 3), order='F')
# x = np.pad(b, ((2, 2), (0, 0)), 'constant', constant_values=(0, 0))

# b[:, :2] = np.array([[100, 100], [100, 100]])
# c = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
from real_traces_generator import real_trace_container_random

# b = np.array([[8, 9, 10], [-1, 0, 0]])
# a = np.array([[1, 2, 3], [4, 5, 6]])

# x = pearsonr(a, a)
# b = np.linalg.norm(a, np.inf, axis=1)
# x = np.corrcoef(a)
# xx = np.corrcoef([100, 2, 3], [0, -1, -3])
# xxx = np.corrcoef([100, 2, 3], [-100, 50, 9])
# xxxx = np.corrcoef([0, -1, -3], [-100, 50, 9])
# print(np.corrcoef(3, 4))
# a = np.random.normal(loc=3, scale=2, size=1000)
# b = moment(a, moment=4)
# moments = describe(a)
# bb = moments.kurtosis
# for x, y in ((1, 2), (3, 4)):
#     print(x, y)
# ccc = c.T[:, :, None] * c.T[:, None]

# t = np.array(c[:, 0], ndmin=2).T * np.array(c[:, 0], ndmin=2)
# import os
#
# from real_traces_generator import real_trace_container_random
#
# data_path = './results_attack/cpa_real'
# filename = os.listdir(data_path)
# df = pd.read_excel(os.sep.join([data_path, 'along_time', 'tables', 'cpa.xlsx']), header=None)
# data = np.array(df)[1:, 1:]
# res = unravel_index(data.argmax(), data.shape)

# container, t_info = real_trace_container_random(
#     dataset_path='/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx',
#     num_traces=2000,
#     t_start=0,
#     t_end=1262)

# pl = container[:].values['plaintexts']
# xx = container[:].values
# for x in xx:
#     print(x['plaintexts'])
# cp = container[:].values['ciphertexts']
# pl = pl[0, 0]
# cp = cp[0, 0]
#
# # res = np.zeros(1000)
# for k in range(256):
#     if pl ^ k == cp:
#         print(k)
# b = np.array(c.flatten(), ndmin=2).T
# # c = np.reshape(b, (2, 5, 5))
# # # xx = (c ^ 0) & 1
# # b = sbox[c]
# # d = sbox[c[2, 1]]
# d = np.unpackbits(b, axis=1)
# x = (d == 1).sum(axis=1)
# x = eigvalsh(c)

# to_mat = np.reshape(c.flatten(order='F'), (9, -1), order='F')
# updated_hist_counts = np.pad(to_mat, ((2, 2), (0, 0)), 'constant',
#                              constant_values=(0, 0))
# cxx = updated_hist_counts.flatten(order='F')
# def _calc_chi2score(chi2_table):
#     n = np.sum(chi2_table)
#     col_sum = np.sum(chi2_table, axis=0, keepdims=True)
#     row_sum = np.sum(chi2_table, axis=1, keepdims=True)
#     expected_freq = np.dot(row_sum, col_sum) / n
#     tmp1 = (chi2_table - expected_freq) ** 2
#     tmp2 = np.divide(tmp1, expected_freq, out=np.zeros_like(tmp1), where=expected_freq != 0)
#     chi_score = np.sum(tmp2)
#     return chi_score
#
# a = np.array([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
# a = np.array([[1, 3, 5, 2, 10, 2, 1], [10, 1, 5, 9, 3, 7, 8]])
# print(_calc_chi2score(a))
# a = np.random.randn(3, 2, 6)
# b = [a, a]
# c = np.array(b)
# print(np.linalg.eig(a)[0])
# print(np.linalg.eigvalsh(a))
# d = np.sum(c, axis=(1, 2))
# left_padding = np.zeros((3, 2, 2, 1))
# right_padding = np.zeros((3, 2, 2, 1))
# # a = np.array([[[1, 2, 4], [3, 4, 1]], [[2, 2, 0], [2, 1, 8]]])
# b = np.reshape(a, (3, 2, 2, 3))
# updated_hist_count = np.concatenate((left_padding, b, right_padding), axis=-1)
# hist_counts = np.reshape(updated_hist_count,
#                          (3, 2, 10))
# hist_counts = np.transpose(a, (0, 2, 1))
# cs = np.sum(a, axis=2, keepdims=True)
# rs = np.sum(a, axis=1, keepdims=True)
# b = cs @ rs
# print(b[1])
# b = np.reshape(a, (3, 2, 3), order='C')
# b = np.array(a[:2, 0], ndmin=2)
# a = np.array([1, 2, 3, 4])
# b = np.array([2, 1, 6, 7])
# xx = np.corrcoef(a, b)
# print(np.sort(data)[::-1])
# a = [[1, 2, 3, 4], 2, 3, 4]
# b = a
# c = a.copy()
# a[0][0] = 10
data_path = './results_attack/graph_attack/ascad_v5_5_corr_mb'
filenames = os.listdir(data_path)
res = []
for filename in filenames:
    df = pd.read_excel(os.sep.join([data_path, filename, 'along_time', 'tables', 'graph_attack_aio.xlsx']), header=None)
    data = np.array(df)[1:, 1:]
    # res = np.argmin(data, axis=0)
    if filename.split('_#')[-1] == 'trace_7.1k':
        # res.append([filename.split('_#')[1], np.argmax(data, axis=0)])
        if np.argmax(data, axis=0) == 224:
            res.append(int(filename.split('_#')[1].split('_')[1]))
    # print(sorted(res, key=lambda x: x[0]))
res.sort()
print(res)
# data_path = './results_attack/yzs1/'
# df = pd.read_excel(os.sep.join([data_path, 'along_time', 'tables', 'graph_attack_aio.xlsx']), header=None)
# data = np.array(df)[1:, 1:]
# print(np.argmax(data, axis=0))
