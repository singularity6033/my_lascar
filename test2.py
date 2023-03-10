from math import inf

from scipy.stats import norm, bernoulli, pearsonr
from Lib_SCA.lascar import SimulatedPowerTraceFixedRandomContainer
import numpy as np
from Lib_SCA.lascar import hamming
from Lib_SCA.lascar.tools.aes import sbox

# a = np.array([[1, 2], [4, 5]])
# b = np.linalg.norm(a, ord=inf, axis=0, keepdims=True)
from configs.simulation_configs import fixed_random_traces

container = SimulatedPowerTraceFixedRandomContainer(config_params=fixed_random_traces)
xx = container[0].value
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

# trace_idx =
from Lib_SCA.hdf5_files_import import read_hdf5_proj
import matplotlib.pyplot as plt

"""data file location"""
proj_path = '/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx'
'''the number of traces to read'''
num_trcs = 1000
'''the starting sample point in a trace'''
pnt_srt = 0
'''the ending sample point in a trace'''
pnt_end = 100
trcs, plts, cpts = read_hdf5_proj(database_file=proj_path, idx_srt=0, idx_end=num_trcs, start=pnt_srt, end=pnt_end,
                                  load_trcs=True, load_plts=True, load_cpts=True)
