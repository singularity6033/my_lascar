import collections
import os
from math import inf

import pandas as pd
from numpy import unravel_index
from scipy.stats import norm, bernoulli, pearsonr, moment, describe, ks_2samp
from sklearn.preprocessing import OneHotEncoder

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
# a = np.random.randn(2, 5, 5)
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

b = np.array([[8, 9, 10], [-1, 0, 0]])
a = np.array([[1, 2, 3], [4, 5, 6]])


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

container, t_info = real_trace_container_random(
    dataset_path='/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx',
    num_traces=2000,
    t_start=0,
    t_end=1262)

# pl = container[:].values['plaintexts']
xx = container[:].values
for x in xx:
    print(x['plaintexts'])
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

# xx = np.corrcoef(a, b)
# print(np.sort(data)[::-1])
# data_path = './results_attack/graph_attack/fr_v1/#gt_dist_#dmode_edit_#trace_50k'
# df = pd.read_excel(os.sep.join([data_path, 'along_time', 'tables', 'graph_attack_aio.xlsx']), header=None)
# data = np.array(df)[1:, 1:]
# res = np.argmin(data, axis=1)
