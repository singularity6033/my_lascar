from tqdm import tqdm

from Lib_SCA.config_extractor import JSONConfig
from configs.evaluation_configs import graph_distance_test_along_time_psr_config, \
    graph_distance_test_along_time_amp_config
from configs.simulation_configs import fixed_random_traces
from Lib_SCA.lascar import SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import SingleOnePlotOutputMethod
from Lib_SCA.lascar import Session, GraphDistanceEngine_AlongTime
from real_traces_generator import real_trace_container


def graph_distance_test_along_time_psr(params, trace_params, output_path):
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

    graph_distance_engine = GraphDistanceEngine_AlongTime(params['engine_name'],
                                                          partition_function,
                                                          type='psr',
                                                          time_delay=params['time_delay'],
                                                          dim=params['dim'],
                                                          sampling_interval=params['sampling_interval'],
                                                          optimal=False,
                                                          measurement=params['measurement'],
                                                          distance_type=params['distance_type'],
                                                          num_bins=params['num_bins'])

    # We choose here to plot the resulting curve
    session = Session(container, engine=graph_distance_engine,
                      output_method=SingleOnePlotOutputMethod(figure_params=params['figure_params'],
                                                              output_path=output_path,
                                                              filename=params['engine_name'],
                                                              display=False),
                      output_steps=params['batch_size'])

    session.run(batch_size=params['batch_size'])

    del graph_distance_engine
    del session


def graph_distance_test_along_time_amp(params, trace_params, output_path):
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

    graph_distance_engine = GraphDistanceEngine_AlongTime(params['engine_name'],
                                                          partition_function,
                                                          type='amp',
                                                          num_of_amp_groups=params['num_of_amp_groups'],
                                                          num_of_moments=params['num_of_moments'],
                                                          measurement=params['measurement'],
                                                          distance_type=params['distance_type'],
                                                          num_bins=params['num_bins'])

    # We choose here to plot the resulting curve
    session = Session(container, engine=graph_distance_engine,
                      output_method=SingleOnePlotOutputMethod(figure_params=params['figure_params'],
                                                              output_path=output_path,
                                                              filename=params['engine_name'],
                                                              display=False),
                      output_steps=params['batch_size'])

    session.run(batch_size=params['batch_size'])

    del graph_distance_engine
    del session


if __name__ == '__main__':
    trace_info = fixed_random_traces
    '''
    graph_distance_test_along_time_psr
    '''
    gdt_psr_params = graph_distance_test_along_time_psr_config
    # json config file generation
    json_config = JSONConfig('graph_distance_test_along_time_psr_v2')
    # 500k
    for m_number_of_traces in [1000, 5000, 10000, 50000, 100000, 250000]:
        for time_delay in [2, 4]:
            for dim in [50, 100]:
                gdt_psr_params['num_traces'] = m_number_of_traces
                gdt_psr_params['time_delay'] = time_delay
                gdt_psr_params['dim'] = dim
                gdt_psr_params['_id'] = '#td_' + str(time_delay) + '_#dim' + str(dim) + '_#trace_' + str(
                    m_number_of_traces // 1000) + 'k'
                json_config.generate_config(gdt_psr_params)

    dict_list = json_config.get_config()
    for i, dict_i in tqdm(enumerate(dict_list)):
        print('[INFO] Processing #', i)
        graph_distance_test_along_time_psr(dict_i,
                                           trace_info,
                                           output_path='./results/graph_distance_test_along_time_psr_v2/' + dict_i[
                                               '_id'])

    # '''
    # graph_distance_test_along_time_amp
    # '''
    # gdt_amp_params = graph_distance_test_along_time_amp_config
    # # json config file generation
    # json_config = JSONConfig('graph_distance_test_along_time_amp_v2')
    # # 500k
    # for m_number_of_traces in [1000, 5000, 10000, 50000, 100000, 250000]:
    #     for num_of_amp_groups in [5, 10, 20]:
    #         for num_of_moments in range(4, 6):
    #             gdt_amp_params['num_traces'] = m_number_of_traces
    #             gdt_amp_params['num_of_amp_groups'] = num_of_amp_groups
    #             gdt_amp_params['num_of_moments'] = num_of_moments
    #             gdt_amp_params['_id'] = '#group_' + str(num_of_amp_groups) + \
    #                                     '_#moments_' + str(num_of_moments) + \
    #                                     '_#trace_' + str(m_number_of_traces // 1000) + 'k'
    #             json_config.generate_config(gdt_amp_params)
    #
    # dict_list = json_config.get_config()
    # for i, dict_i in tqdm(enumerate(dict_list)):
    #     print('[INFO] Processing #', i)
    #     graph_distance_test_along_time_amp(dict_i,
    #                                        trace_info,
    #                                        output_path='./results/graph_distance_test_along_time_amp_v2/' + dict_i[
    #                                            '_id'])
