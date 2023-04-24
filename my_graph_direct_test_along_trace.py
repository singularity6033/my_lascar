"""
In this script, we show how to perform side-channel characterisation using Welch's T-test to study the behaviour of an Aes Sbox

The characterisation is made with the TTestEngine
Its constructor needs a partition function, which will separate leakages into two classes.

"""
from tqdm import tqdm

from Lib_SCA.config_extractor import JSONConfig
from configs.evaluation_configs import graph_direct_test_along_trace_config
from configs.simulation_configs import fixed_random_traces
from Lib_SCA.lascar import SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import SingleOnePlotOutputMethod
from Lib_SCA.lascar import Session, GraphTestEngine_AlongTrace
from real_traces_generator import real_trace_container


def graph_direct_test_along_trace(params, trace_params, output_path):
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

    graph_test_trace_engine = GraphTestEngine_AlongTrace(params['engine_name'],
                                                         partition_function)
    # We choose here to plot the resulting curve
    session = Session(container, engine=graph_test_trace_engine,
                      output_method=SingleOnePlotOutputMethod(figure_params=params['figure_params'],
                                                              output_path=output_path,
                                                              filename=params['engine_name'],
                                                              display=False))
    session.run(batch_size=params['batch_size'])

    del graph_test_trace_engine
    del session


if __name__ == '__main__':
    # graph_direct_test_along_trace(graph_direct_test_along_trace_config, fixed_random_traces,
    #                               output_path='./results/yzs')
    trace_info = fixed_random_traces

    '''
    graph_direct_test_along_trace
    '''
    gt_trace_params = graph_direct_test_along_trace_config
    # json config file generation
    json_config = JSONConfig('graph_direct_test_along_trace_v1')
    # 500k
    for m_number_of_traces in [1000, 5000, 10000, 50000, 100000, 250000]:
        for m_bs in [10, 20, 50, 100, 500, 1000]:
            gt_trace_params['num_traces'] = m_number_of_traces
            gt_trace_params['batch_size'] = m_bs
            gt_trace_params['_id'] = '#bs_' + str(m_bs) + '_#trace_' + str(m_number_of_traces // 1000) + 'k'
            json_config.generate_config(gt_trace_params)

    # get json config file
    dict_list = json_config.get_config()
    for i, dict_i in tqdm(enumerate(dict_list)):
        print('[INFO] Processing #', i)
        graph_direct_test_along_trace(dict_i, trace_info,
                                      output_path='./results/graph_direct_test_along_trace_v1/' + dict_i['_id'])
