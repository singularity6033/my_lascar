"""
In this script, we show how to perform side-channel characterisation using Welch's T-test to study the behaviour of an Aes Sbox

The characterisation is made with the TTestEngine
Its constructor needs a partition function, which will separate leakages into two classes.

"""
from tqdm import tqdm

from Lib_SCA.config_extractor import JSONConfig
from configs.evaluation_configs import graph_direct_test_attack_psr_config, graph_direct_test_attack_amp_config
from configs.simulation_configs import normal_simulated_traces
from Lib_SCA.lascar import SimulatedPowerTraceContainer
from Lib_SCA.lascar import NonIncrementalOutputMethod
from Lib_SCA.lascar import Session, GraphTestEngine_Attack
from Lib_SCA.lascar.tools.aes import sbox
from real_traces_generator import real_trace_container_random


def graph_direct_test_attack_psr(params, trace_params, output_path):
    container = None
    if params['mode'] == 'normal':
        container = SimulatedPowerTraceContainer(config_params=trace_params)
        # attack_byte = container.idx_exp[0]
        # attack_time = container.attack_sample_point
    elif params['mode'] == 'real':
        container, t_info = real_trace_container_random(dataset_path=params['dataset_path'],
                                                        num_traces=params['num_traces'],
                                                        t_start=0,
                                                        t_end=100)

    attack_byte = params['attack_byte']

    # selection attack regions along time axis
    # container.leakage_section = params['attack_range']

    def selection_function(value, guess, ab=attack_byte):
        # LSB
        return sbox[value["plaintexts"][ab] ^ guess] & 1

    guess_range = range(params['no_of_key_guesses'])

    graph_direct_test_attack_psr_engine = GraphTestEngine_Attack(params['engine_name'],
                                                                 selection_function,
                                                                 guess_range,
                                                                 type='psr',
                                                                 time_delay=params['time_delay'],
                                                                 dim=params['dim'],
                                                                 sampling_interval=params['sampling_interval'],
                                                                 optimal=False,
                                                                 measurement=params['measurement'])
    session = Session(container,
                      engine=graph_direct_test_attack_psr_engine,
                      output_method=NonIncrementalOutputMethod(
                          figure_params=params['figure_params'],
                          output_path=output_path),
                      output_steps=params['batch_size']
                      )

    session.run(batch_size=params['batch_size'])

    del graph_direct_test_attack_psr_engine
    del session


def graph_direct_test_attack_amp(params, trace_params, output_path):
    container = None
    if params['mode'] == 'normal':
        container = SimulatedPowerTraceContainer(config_params=trace_params)
        # attack_byte = container.idx_exp[0]
        # attack_time = container.attack_sample_point
    elif params['mode'] == 'real':
        container, t_info = real_trace_container_random(dataset_path=params['dataset_path'],
                                                        num_traces=params['num_traces'],
                                                        t_start=0,
                                                        t_end=100)

    attack_byte = params['attack_byte']

    # selection attack regions along time axis
    # container.leakage_section = params['attack_range']

    def selection_function(value, guess, ab=attack_byte):
        # LSB
        return sbox[value["plaintexts"][ab] ^ guess] & 1

    guess_range = range(params['no_of_key_guesses'])

    graph_direct_test_attack_amp_engine = GraphTestEngine_Attack(params['engine_name'],
                                                                 selection_function,
                                                                 guess_range,
                                                                 type='amp',
                                                                 num_of_amp_groups=params['num_of_amp_groups'],
                                                                 num_of_moments=params['num_of_moments'],
                                                                 measurement=params['measurement'])
    session = Session(container,
                      engine=graph_direct_test_attack_amp_engine,
                      output_method=NonIncrementalOutputMethod(
                          figure_params=params['figure_params'],
                          output_path=output_path),
                      output_steps=params['batch_size']
                      )

    session.run(batch_size=params['batch_size'])

    del graph_direct_test_attack_amp_engine
    del session


if __name__ == '__main__':
    # graph_direct_test_attack_psr(graph_direct_test_attack_psr_config, normal_simulated_traces,
    #                              output_path='./results/yzs')
    # graph_direct_test_attack_amp(graph_direct_test_attack_amp_config, normal_simulated_traces,
    #                              output_path='./results/yzs')
    trace_info = normal_simulated_traces
    '''
    graph_direct_test_attack_psr
    '''
    # gta_psr_params = graph_direct_test_attack_psr_config
    #
    # # json config file generation
    # json_config = JSONConfig('graph_direct_test_attack_psr_v1')
    # # 500k
    # for m_number_of_traces in [1000, 5000, 10000, 50000, 100000, 250000, 500000, 1000000, 2000000]:
    #     for time_delay in [1, 2, 3]:
    #         for dim in [10, 50, 75, 100]:
    #             gta_psr_params['num_traces'] = m_number_of_traces
    #             gta_psr_params['time_delay'] = time_delay
    #             gta_psr_params['dim'] = dim
    #             gta_psr_params['_id'] = '#td_' + str(time_delay) + '_#dim' + str(dim) + '_#trace_' + str(
    #                 m_number_of_traces // 1000) + 'k'
    #             json_config.generate_config(gta_psr_params)
    #
    # dict_list = json_config.get_config()
    # for i, dict_i in tqdm(enumerate(dict_list)):
    #     print('[INFO] Processing #', i)
    #     graph_direct_test_attack_psr(dict_i, trace_info,
    #                                  output_path='./results/graph_direct_test_attack_psr_v1/' + dict_i['_id'])

    '''
    graph_direct_test_attack_amp
    '''
    gta_amp_params = graph_direct_test_attack_amp_config
    # json config file generation
    json_config = JSONConfig('graph_direct_test_attack_amp_v1')
    # 500k
    for m_number_of_traces in [1000, 5000, 10000, 50000, 100000, 250000, 500000, 1000000, 2000000]:
        for num_of_amp_groups in [5, 10, 20, 50, 100]:
            for num_of_moments in range(4, 6):
                gta_amp_params['num_traces'] = m_number_of_traces
                gta_amp_params['num_of_amp_groups'] = num_of_amp_groups
                gta_amp_params['num_of_moments'] = num_of_moments
                gta_amp_params['_id'] = '#group_' + str(num_of_amp_groups) + \
                                        '_#moments_' + str(num_of_moments) + \
                                        '_#trace_' + str(m_number_of_traces // 1000) + 'k'
                json_config.generate_config(gta_amp_params)

    # get json config file
    dict_list = json_config.get_config()
    for i, dict_i in tqdm(enumerate(dict_list)):
        print('[INFO] Processing #', i)
        graph_direct_test_attack_amp(dict_i, trace_info,
                                     output_path='./results/graph_direct_test_attack_amp_v1/' + dict_i['_id'])
