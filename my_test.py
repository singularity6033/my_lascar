import numpy as np

from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
import matplotlib.pyplot as plt

# container = SimulatedPowerTraceContainer('normal_simulated_traces.yaml')
container = SimulatedPowerTraceFixedRandomContainer('fixed_random_traces.yaml')

container.plot_traces([0, container.number_of_traces])
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
