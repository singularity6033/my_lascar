"""
In this script, we show how to perform side-channel characterisation using Welch's T-test to study the behaviour of an Aes Sbox

The characterisation is made with the TTestEngine
Its constructor needs a partition function, which will separate leakages into two classes.

"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
from tqdm import tqdm

from Lib_SCA.config_extractor import YAMLConfig, JSONConfig
from Lib_SCA.configs.evaluation_configs import graph_test_config
from Lib_SCA.configs.simulation_configs import fixed_random_traces
from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import Single_Result_OutputMethod, Incremental_Batch_OutputMethod
from Lib_SCA.lascar import Session, GraphTestEngine, GraphMIEngine, GraphDistanceEngine


# from real_traces_generator import real_trace_generator


def graph_based_test(params, trace_params):
    if params['mode'] == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_params=trace_params)
    elif params['mode'] == 'real':
        pass

    def partition_function(value):
        # partition_function must take 1 argument: the value returned by the container at each trace
        # fix and random sets have already been partitioned in container
        return int(value["trace_idx"] % 2 != 0)

    graph_test_engine = GraphTestEngine(params['engine_name'],
                                        partition_function,
                                        time_delay=2,
                                        dim=3)
    # output_path = trace_params['_id']
    # We choose here to plot the resulting curve
    session = Session(container, engine=graph_test_engine,
                      output_method=Incremental_Batch_OutputMethod(figure_params=params['figure_params'],
                                                                   output_path='',
                                                                   display=False),
                      output_steps=params['batch_size'])
    session.run(batch_size=params['batch_size'])


def graph_based_mi(params, trace_params):
    if params['mode'] == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_params=trace_params)
    elif params['mode'] == 'real':
        pass

    def partition_function(value):
        # partition_function must take 1 argument: the value returned by the container at each trace
        # fix and random sets have already been partitioned in container
        return int(value["trace_idx"] % 2 != 0)

    graph_test_engine = GraphMIEngine(params['engine_name'],
                                      partition_function,
                                      time_delay=2,
                                      dim=3)
    output_path = trace_params['_id']

    # We choose here to plot the resulting curve
    session = Session(container, engine=graph_test_engine,
                      output_method=Single_Result_OutputMethod(figure_params=params['figure_params'],
                                                               output_path=output_path,
                                                               display=False))
    session.run(batch_size=params['batch_size'])


def graph_based_distance(params, trace_params):
    if params['mode'] == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_params=trace_params)
    elif params['mode'] == 'real':
        pass

    def partition_function(value):
        # partition_function must take 1 argument: the value returned by the container at each trace
        # fix and random sets have already been partitioned in container
        return int(value["trace_idx"] % 2 != 0)

    graph_test_engine = GraphDistanceEngine(params['engine_name'],
                                            partition_function,
                                            time_delay=2,
                                            dim=3)

    # We choose here to plot the resulting curve
    session = Session(container, engine=graph_test_engine)
    session.run(batch_size=params['batch_size'])

    del session


if __name__ == '__main__':
    # gt_params = graph_test_config
    # trace_info = fixed_random_traces
    # # json config file generation
    # json_config = JSONConfig('graph_test_1')
    # # 10k, 50k, 200k, 500k, 1000k
    # for m_noise_sigma_el in [0, 0.5, 1]:
    #     for shuffle_state in [True, False]:
    #         for shift_state in [True, False]:
    #             for m_masking_bytes in [0, 1]:
    #                 trace_info['noise_sigma_el'] = m_noise_sigma_el
    #                 trace_info['shuffle'] = shuffle_state
    #                 trace_info['shift'] = shift_state
    #                 trace_info['number_of_masking_bytes'] = m_masking_bytes
    #                 trace_info['_id'] = '#mask_' + str(m_masking_bytes) + '_el_' + str(m_noise_sigma_el) + \
    #                                     '_#shuffle_' + str(shuffle_state) + '_#shift_' + str(shift_state)
    #                 json_config.generate_config(trace_info)
    # #
    # # get json config file
    # dict_list = json_config.get_config()
    # for i, dict_i in tqdm(enumerate(dict_list)):
    #     print('[INFO] Processing #', i)
    #     graph_based_test(gt_params, dict_i)
    #     # graph_based_mi(gt_params, dict_i)

    # graph_based_mi(graph_test_config, fixed_random_traces)
    graph_based_test(graph_test_config, fixed_random_traces)
    # graph_based_distance(graph_test_config, fixed_random_traces)
