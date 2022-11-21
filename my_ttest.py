"""
In this script, we show how to perform side-channel characterisation using Welch's T-test to study the behaviour of an Aes Sbox

The characterisation is made with the TTestEngine
Its constructor needs a partition function, which will separate leakages into two classes.

"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

from Lib_SCA.config_extractor import TraceConfig
from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer, \
    Single_Result_OutputMethod
from Lib_SCA.lascar import Session, TTestEngine
from real_traces_generator import real_trace_generator


def t_test(config_name):
    ttest_params = TraceConfig().get_config(config_name)
    if ttest_params['mode'] == 'fix_random':
        container_params = TraceConfig().get_config('fixed_random_traces.yaml')
        container = SimulatedPowerTraceFixedRandomContainer(config_params=container_params)
    elif ttest_params['mode'] == 'real':
        pass

    def partition_function(value):
        # partition_function must take 1 argument: the value returned by the container at each trace
        # fix and random sets have already been partitioned in container
        return int(value["trace_idx"] % 2 == 0)

    ttest_engine = TTestEngine(ttest_params['engine_name'], partition_function)

    # We choose here to plot the resulting curve
    session = Session(container,
                      engine=ttest_engine,
                      output_method=Single_Result_OutputMethod(figure_params=ttest_params['figure_params'],
                                                               output_path='./plots/ttest.png'))
    session.run(batch_size=ttest_params['batch_size'])

    # comparison with Scipy built-in function
    # results = ttest_engine.finalize()
    # plt.figure(1)
    # plt.plot(results.T)
    # equal_var = len(container) % 2 == 0
    # real_leakages = container[:len(container)].leakages
    # results_v = ttest_ind(real_leakages[1::2], real_leakages[::2], axis=0, equal_var=equal_var).statistic
    # plt.plot(results_v, 'o')
    # plt.legend(['from lascar', 'from scipy'])
    # plt.show()


if __name__ == '__main__':
    t_test(config_name='t_test.yaml')
