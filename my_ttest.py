"""
In this script, we show how to perform side-channel characterisation using Welch's T-test to study the behaviour of an Aes Sbox

The characterisation is made with the TTestEngine
Its constructor needs a partition function, which will separate leakages into two classes.

"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

from Lib_SCA.lascar import Session, MatPlotLibOutputMethod, SimulatedPowerTraceFixedRandomContainer, TTestEngine


def t_test(config_name, engine_name='ttest', batch_size=2500):
    container = SimulatedPowerTraceFixedRandomContainer(config_name)

    def partition_function(value):
        # partition_function must take 1 argument: the value returned by the container at each trace
        # fix and random sets have already been partitioned in container
        return int(value["trace_idx"] % 2 == 0)

    ttest_engine = TTestEngine(engine_name, partition_function)

    # We choose here to plot the resulting curve
    plot_output = MatPlotLibOutputMethod(ttest_engine)
    session = Session(container, output_method=plot_output)
    session.add_engine(ttest_engine)
    session.run(batch_size=batch_size)

    # comparison with Scipy built-in function
    results = ttest_engine.finalize()
    plt.figure(1)
    plt.plot(results.T)
    equal_var = len(container) % 2 == 0
    real_leakages = container[:len(container)].leakages
    results_v = ttest_ind(real_leakages[1::2], real_leakages[::2], axis=0, equal_var=equal_var).statistic
    plt.plot(results_v, 'o')
    plt.legend(['from lascar', 'from scipy'])
    plt.show()


if __name__ == '__main__':
    t_test(config_name='fixed_random_traces.yaml', engine_name='ttest', batch_size=2500)
