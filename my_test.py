from math import inf

import numpy as np
from scipy.stats import mvn, multivariate_normal
from Lib_SCA.lascar import numerical_success_rate
from sklearn.neighbors import KernelDensity

from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
import matplotlib.pyplot as plt

# container = SimulatedPowerTraceContainer('normal_simulated_traces.yaml')
# container = SimulatedPowerTraceFixedRandomContainer('fixed_random_traces.yaml')
#
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
x = np.random.normal(0, 1.0, (1000, 2))
bd = np.sum(x, axis=0)
bdd = np.array(bd, ndmin=2)
# kde = KernelDensity(kernel='epanechnikov', bandwidth=0.2).fit(x)
# y = kde.score_samples(x)
# y1 = np.exp(y)
# plt.figure(1)
# plt.plot(x, y1)
# plt.show()
