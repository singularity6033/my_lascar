from tqdm import tqdm

from Lib_SCA.config_extractor import JSONConfig
from configs.evaluation_configs import graph_distance_test_along_trace_config
from configs.simulation_configs import fixed_random_traces
from Lib_SCA.lascar import SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import SingleOnePlotOutputMethod
from Lib_SCA.lascar import Session, GraphDistanceEngine_AlongTrace
from real_traces_generator import real_trace_container


def graph_distance_test_along_trace(params, trace_params, output_path):
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

    graph_based_distance_trace_engine = GraphDistanceEngine_AlongTrace(params['engine_name'],
                                                                       partition_function,
                                                                       distance_type=params['distance_type'],
                                                                       num_bins=params['num_bins'])
    # We choose here to plot the resulting curve
    session = Session(container, engine=graph_based_distance_trace_engine,
                      output_method=SingleOnePlotOutputMethod(figure_params=params['figure_params'],
                                                              output_path=output_path,
                                                              filename=params['engine_name'],
                                                              display=False))
    session.run(batch_size=params['batch_size'])

    del graph_based_distance_trace_engine
    del session


if __name__ == '__main__':
    trace_info = fixed_random_traces

    '''
    graph_distance_test_along_trace
    '''
    gdt_trace_params = graph_distance_test_along_trace_config
    # graph_distance_test_along_trace(gdt_trace_params, trace_info, output_path='./results/yzs')

    # json config file generation
    json_config = JSONConfig('graph_distance_test_along_trace_v1')
    # 500k
    for m_number_of_traces in [1000, 5000, 10000, 50000, 100000, 250000]:
        for m_bs in [10, 20, 50, 100, 500, 1000]:
            gdt_trace_params['num_traces'] = m_number_of_traces
            gdt_trace_params['batch_size'] = m_bs
            gdt_trace_params['_id'] = '#bs_' + str(m_bs) + '_#trace_' + str(m_number_of_traces // 1000) + 'k'
            json_config.generate_config(gdt_trace_params)

    dict_list = json_config.get_config()
    for i, dict_i in tqdm(enumerate(dict_list)):
        print('[INFO] Processing #', i)
        graph_distance_test_along_trace(dict_i,
                                        trace_info,
                                        output_path='./results/graph_distance_test_along_trace_v1/' + dict_i['_id'])