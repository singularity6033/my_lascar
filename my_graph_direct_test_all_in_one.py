"""
In this script, we show how to perform side-channel characterisation using Welch's T-test to study the behaviour of an Aes Sbox

The characterisation is made with the TTestEngine
Its constructor needs a partition function, which will separate leakages into two classes.

"""
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from Lib_SCA.config_extractor import JSONConfig
from configs.evaluation_configs import graph_direct_test_aio_config
from configs.simulation_configs import fixed_random_traces
from Lib_SCA.lascar import SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import SingleOnePlotOutputMethod
from Lib_SCA.lascar import Session, GraphTestingEngine_AllInOne_Correlation, GraphTestingEngine_AllInOne_Distance
from real_traces_generator import real_trace_container


def graph_direct_test_aio(params, trace_params, output_path):
    container = None
    if params['mode'] == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_params=trace_params)
    elif params['mode'] == 'real':
        container, t_info = real_trace_container(dataset_path=params['dataset_path'],
                                                 num_traces=params['num_traces'],
                                                 t_start=0,
                                                 t_end=1262)

    def partition_function(value):
        # partition_function must take 1 argument: the value returned by the container at each trace
        # fix and random sets have already been partitioned in container
        return int(value["trace_idx"] % 2 == 0)

    if params['type'] == 'corr':
        graph_test_engine = GraphTestingEngine_AllInOne_Correlation(params['engine_name'], partition_function,
                                                                    k=params['k'])
    elif params['type'] == 'dist':
        graph_test_engine = GraphTestingEngine_AllInOne_Distance(params['engine_name'], partition_function,
                                                                 k=params['k'])

    # We choose here to plot the resulting curve
    session = Session(container,
                      engine=graph_test_engine,
                      output_method=SingleOnePlotOutputMethod(figure_params=params['figure_params'],
                                                              output_path=output_path,
                                                              filename=params['engine_name'],
                                                              display=False))
    session.run(batch_size=params['batch_size'])

    del graph_test_engine
    del session


if __name__ == '__main__':
    graph_direct_test_aio(graph_direct_test_aio_config, fixed_random_traces, output_path='./results/yzs')
    # num_traces = [500, 750, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    # num_clusters = [2, 3, 5]
    # trace_info = fixed_random_traces
    # '''
    # graph_direct_test_all_in_one
    # '''
    # gt_aio_params = graph_direct_test_aio_config
    # # json config file generation
    # json_config = JSONConfig('graph_direct_test_all_in_one_dist_v1')
    # # 500k
    # for k in num_clusters:
    #     for m in num_traces:
    #         gt_aio_params['num_traces'] = m
    #         gt_aio_params['k'] = k
    #         gt_aio_params['_id'] = '#r' + str(k) + '_#trace_' + str(m)
    #         json_config.generate_config(gt_aio_params)
    #
    # dict_list = json_config.get_config()
    # for i, dict_i in tqdm(enumerate(dict_list)):
    #     print('[INFO] Processing #', i)
    #     graph_direct_test_aio(dict_i,
    #                           trace_info,
    #                           output_path='./results/graph_direct_test_all_in_one_dist_v1/' + dict_i['_id'])
    #
    # from Lib_SCA.lascar.engine.graph_engine_direct_test import gamma_diff_save, miu_diff_save, rd_save, epsilon_save
    # data_path = './results/graph_direct_test_all_in_one_dist_v1_aux'
    # if not os.path.exists(data_path):
    #     os.mkdir(data_path)
    # len_k = len(num_clusters)
    # len_traces = len(num_traces)
    #
    # with pd.ExcelWriter(os.sep.join([data_path, 'miu_diff.xlsx']), engine='xlsxwriter') as writer:
    #     for i, type in enumerate(['fixed', 'random']):
    #         df = pd.DataFrame(np.array(miu_diff_save[i]).reshape((len_k, len_traces)), columns=num_traces)
    #         df.to_excel(writer, sheet_name=type)
    #
    # with pd.ExcelWriter(os.sep.join([data_path, 'rd.xlsx']), engine='xlsxwriter') as writer:
    #     for i, type in enumerate(['fixed', 'random']):
    #         df = pd.DataFrame(np.array(rd_save[i]).reshape((len_k, len_traces)), columns=num_traces)
    #         df.to_excel(writer, sheet_name=type)
    #
    # with pd.ExcelWriter(os.sep.join([data_path, 'epsilon.xlsx']), engine='xlsxwriter') as writer:
    #     for i, type in enumerate(['fixed', 'random']):
    #         df = pd.DataFrame(np.array(epsilon_save[i]).reshape((len_k, len_traces)), columns=num_traces)
    #         df.to_excel(writer, sheet_name=type)
    #
    # with pd.ExcelWriter(os.sep.join([data_path, 'gamma_diff.xlsx']), engine='xlsxwriter') as writer:
    #     df = pd.DataFrame(np.array(gamma_diff_save, ndmin=2).T.reshape((len_k, len_traces)), columns=num_traces)
    #     df.to_excel(writer)
