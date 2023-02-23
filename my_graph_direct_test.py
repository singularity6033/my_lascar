"""
In this script, we show how to perform side-channel characterisation using Welch's T-test to study the behaviour of an Aes Sbox

The characterisation is made with the TTestEngine
Its constructor needs a partition function, which will separate leakages into two classes.

"""
from tqdm import tqdm

from Lib_SCA.config_extractor import JSONConfig
from configs.evaluation_configs import graph_test_config, graph_test_attack_config, graph_test_trace_based_config
from configs.simulation_configs import normal_simulated_traces, fixed_random_traces
from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import SingleOnePlotOutputMethod, NonIncrementalOutputMethod
from Lib_SCA.lascar import Session, GraphTestEngine, GraphTestEngine_Attack, TraceBasedGraphTestEngine
from Lib_SCA.lascar.tools.aes import sbox


# from real_traces_generator import real_trace_generator


def graph_based_test(params, trace_params, output_path):
    container = None
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
    # We choose here to plot the resulting curve
    session = Session(container,
                      engine=graph_test_engine,
                      output_method=SingleOnePlotOutputMethod(figure_params=params['figure_params'],
                                                              output_path=output_path,
                                                              filename=params['engine_name'],
                                                              display=False),
                      output_steps=params['batch_size'])
    session.run(batch_size=params['batch_size'])


def graph_based_test_attack(params, trace_params, output_path):
    container = None
    if params['mode'] == 'normal':
        container = SimulatedPowerTraceContainer(config_params=trace_params)
    elif params['mode'] == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_params=trace_params)
    elif params['mode'] == 'real':
        pass
        # container = real_trace_generator()

    attack_byte = container.idx_exp[0]
    attack_time = container.attack_sample_point

    # selection attack regions along time axis
    # container.leakage_section = params['attack_range']

    def selection_function(value, guess, ab=attack_byte, at=attack_time):
        # LSB
        return sbox[value["plaintext"][ab][at] ^ guess] & 1

    guess_range = range(params['no_of_key_guesses'])

    graph_direct_test_engine = GraphTestEngine_Attack(params['engine_name'],
                                                      selection_function,
                                                      guess_range,
                                                      time_delay=2,
                                                      dim=3,
                                                      solution=params['idx_of_correct_key_guess'])
    session = Session(container,
                      engine=graph_direct_test_engine,
                      output_method=NonIncrementalOutputMethod(
                          figure_params=params['figure_params'],
                          output_path=output_path),
                      output_steps=params['batch_size']
                      )

    session.run(batch_size=params['batch_size'])

    del graph_direct_test_engine
    del session


def graph_based_test_trace(params, trace_params, output_path):
    container = None
    if params['mode'] == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_params=trace_params)
    elif params['mode'] == 'real':
        pass

    def partition_function(value):
        # partition_function must take 1 argument: the value returned by the container at each trace
        # fix and random sets have already been partitioned in container
        return int(value["trace_idx"] % 2 != 0)

    graph_test_trace_engine = TraceBasedGraphTestEngine(params['engine_name'],
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
    # graph_based_test_trace(graph_test_trace_based_config, fixed_random_traces, output_path='./results/yzs')
    # graph_based_test(graph_test_config, fixed_random_traces, output_path='./results/yzs')
    # graph_based_test_attack(graph_test_attack_config, normal_simulated_traces, output_path='./results/yzs')
    '''
    graph direct test
    '''
    gt_params = graph_test_config
    trace_info = fixed_random_traces
    # json config file generation
    json_config = JSONConfig('graph_based_test_v4')
    # 500k
    for m_number_of_traces in [50000, 100000, 250000, 500000]:
        for m_noise_sigma_el in [0, 0.25, 0.5, 1]:
            for m_masking_bytes in range(10):
                trace_info['number_of_traces'] = m_number_of_traces
                trace_info['noise_sigma_el'] = m_noise_sigma_el
                trace_info['number_of_masking_bytes'] = m_masking_bytes
                trace_info['_id'] = 'el_' + str(m_noise_sigma_el) + \
                                    '_#mask_' + str(m_masking_bytes) + \
                                    '_#trace_' + str(trace_info['number_of_traces'] // 1000) + 'k'
                json_config.generate_config(trace_info)

    dict_list = json_config.get_config()
    for i, dict_i in tqdm(enumerate(dict_list)):
        print('[INFO] Processing #', i)
        graph_based_test(gt_params, dict_i, output_path='./results/graph_based_test_v4/' + dict_i['_id'])

    '''
    trace based graph direct test
    '''
    gt_params = graph_test_trace_based_config
    trace_info = fixed_random_traces
    # json config file generation
    json_config = JSONConfig('graph_based_test_trace_v4')
    # 500k
    for m_number_of_traces in [50000, 100000, 250000, 500000]:
        for m_noise_sigma_el in [0, 0.25, 0.5, 1]:
            for m_masking_bytes in range(10):
                trace_info['number_of_traces'] = m_number_of_traces
                trace_info['noise_sigma_el'] = m_noise_sigma_el
                trace_info['number_of_masking_bytes'] = m_masking_bytes
                trace_info['_id'] = 'el_' + str(m_noise_sigma_el) + \
                                    '_#mask_' + str(m_masking_bytes) + \
                                    '_#trace_' + str(trace_info['number_of_traces'] // 1000) + 'k'
                json_config.generate_config(trace_info)

    # get json config file
    dict_list = json_config.get_config()
    for i, dict_i in tqdm(enumerate(dict_list)):
        print('[INFO] Processing #', i)
        graph_based_test_trace(gt_params, dict_i, output_path='./results/graph_based_test_trace_v4/' + dict_i['_id'])

    '''
    attacked based graph direct test
    '''
    gt_params = graph_test_attack_config
    trace_info = normal_simulated_traces
    # json config file generation
    json_config = JSONConfig('graph_based_test_attack_v4')
    # 10k, 50k, 200k, 500k, 1000k
    for m_number_of_traces in [50000, 100000, 250000, 500000]:
        for m_noise_sigma_el in [0, 0.25, 0.5, 1]:
            for m_masking_bytes in range(10):
                trace_info['number_of_traces'] = m_number_of_traces
                trace_info['noise_sigma_el'] = m_noise_sigma_el
                trace_info['number_of_masking_bytes'] = m_masking_bytes
                trace_info['_id'] = 'el_' + str(m_noise_sigma_el) + \
                                    '_#mask_' + str(m_masking_bytes) + \
                                    '_#trace_' + str(trace_info['number_of_traces'] // 1000) + 'k'
                json_config.generate_config(trace_info)

    # get json config file
    dict_list = json_config.get_config()
    for i, dict_i in tqdm(enumerate(dict_list)):
        print('[INFO] Processing #', i)
        graph_based_test_attack(gt_params, dict_i, output_path='./results/graph_based_test_attack_v4/' + dict_i['_id'])

    # graph_based_test_attack(graph_test_attack_config, normal_simulated_traces, output_path='./results/graph_based_test_attack/')
