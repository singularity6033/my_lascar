from math import inf

from scipy.stats import norm, bernoulli, pearsonr, moment, describe
from Lib_SCA.lascar import SimulatedPowerTraceFixedRandomContainer
import numpy as np
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
# from bisect import bisect_left, bisect_right
# levels = np.arange(0, 10 + (10 - 0) /10, (10 - 0) /10).tolist()
# l = [i for i in range(11)]
# index = bisect_left(l, 1)
# a = np.random.normal(loc=3, scale=2, size=1000)
# b = moment(a, moment=4)
# moments = describe(a)
# bb = moments.kurtosis

