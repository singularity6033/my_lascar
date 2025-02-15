"""
In this script, we show how to perform side-channel characterisation using Welch's T-test to study the behaviour of an Aes Sbox

The characterisation is made with the TTestEngine
Its constructor needs a partition function, which will separate leakages into two classes.

"""
import numpy as np
from scipy.stats import ttest_ind
from tqdm import tqdm

from Lib_SCA.config_extractor import JSONConfig
from Lib_SCA.hdf5_files_import import read_hdf5_proj
from configs.evaluation_configs import t_test_config
from configs.simulation_configs import fixed_random_traces
from Lib_SCA.lascar import SimulatedPowerTraceFixedRandomContainer, TraceBatchContainer
from Lib_SCA.lascar import SingleVectorPlotOutputMethod
from Lib_SCA.lascar import Session, TTestEngine
from real_traces_generator import real_trace_container


def tt_test(params, trace_params, output_path):
    container = None
    if params['mode'] == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_params=trace_params)
    elif params['mode'] == 'real':
        container, _ = real_trace_container(dataset_path=params['dataset_path'],
                                            num_traces=params['num_traces'],
                                            t_start=0,
                                            t_end=1262)

    def partition_function(value):
        # partition_function must take 1 argument: the value returned by the container at each trace
        # fix and random sets have already been partitioned in container
        return int(value["trace_idx"] % 2 == 0)

    ttest_engine = TTestEngine(params['engine_name'], partition_function)

    # We choose here to plot the resulting curve
    session = Session(container,
                      engine=ttest_engine,
                      output_method=SingleVectorPlotOutputMethod(
                          figure_params_along_time=params['figure_params_along_time'],
                          figure_params_along_trace=params['figure_params_along_trace'],
                          output_path=output_path,
                          filename=params['engine_name']
                      ),
                      output_steps=params['batch_size'])
    session.run(batch_size=params['batch_size'])

    del ttest_engine
    del session

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
    # tt_test(t_test_config, fixed_random_traces, output_path='./results/t-test')
    ttest_params = t_test_config
    trace_info = fixed_random_traces
    # json config file generation
    json_config = JSONConfig('ttest_real_v1')

    for m_number_of_traces in [1000, 5000, 10000, 50000, 100000, 250000, 500000, 1000000, 2000000]:
        ttest_params['num_traces'] = m_number_of_traces
        ttest_params['_id'] = '#trace_' + str(m_number_of_traces // 1000) + 'k'
        json_config.generate_config(ttest_params)

    # get json config file
    dict_list = json_config.get_config()
    for i, dict_i in tqdm(enumerate(dict_list)):
        print('[INFO] Processing #', i)
        tt_test(dict_i, trace_info, output_path='./results/ttest_real_v1/' + dict_i['_id'])
