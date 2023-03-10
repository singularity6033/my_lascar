from math import inf

from scipy.stats import norm, bernoulli, pearsonr
import numpy as np
from Lib_SCA.lascar import hamming
from Lib_SCA.lascar.tools.aes import sbox

# a = np.array([[1, 2], [4, 5]])
# b = np.linalg.norm(a, ord=inf, axis=0, keepdims=True)


from Lib_SCA.hdf5_files_import import read_hdf5_proj
import matplotlib.pyplot as plt

"""data file location"""
proj_path = '..//sca_real_data/EM_Sync_TVLA_1M.sx'
'''the number of traces to read'''
num_trcs = 1000
'''the starting sample point in a trace'''
pnt_srt = 0
'''the ending sample point in a trace'''
pnt_end = 100000
trcs, plts, cpts = read_hdf5_proj(database_file=proj_path, idx_srt=0, idx_end=num_trcs, start=pnt_srt, end=pnt_end,
                                  load_trcs=True, load_plts=True, load_cpts=True)
