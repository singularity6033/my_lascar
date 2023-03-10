from tqdm import tqdm

from Lib_SCA.config_extractor import JSONConfig
from configs.evaluation_configs import graph_distance_config, graph_distance_trace_based_config
from configs.simulation_configs import fixed_random_traces
from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import SingleOnePlotOutputMethod, NonIncrementalOutputMethod
from Lib_SCA.lascar import Session, GraphDistanceEngine, TraceBasedGraphDistanceEngine, GraphDistanceEngine_Attack
from Lib_SCA.lascar.tools.aes import sbox


def graph_based_distance(params, trace_params, output_path):
    container = None
    if params['mode'] == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_params=trace_params)
    elif params['mode'] == 'real':
        pass

    def partition_function(value):
        # partition_function must take 1 argument: the value returned by the container at each trace
        # fix and random sets have already been partitioned in container
        return int(value["trace_idx"] % 2 != 0)

    graph_distance_engine = GraphDistanceEngine(params['engine_name'],
                                                partition_function,
                                                time_delay=2,
                                                dim=3,
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


def graph_based_distance_trace(params, trace_params, output_path):
    container = None
    if params['mode'] == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_params=trace_params)
    elif params['mode'] == 'real':
        pass

    def partition_function(value):
        # partition_function must take 1 argument: the value returned by the container at each trace
        # fix and random sets have already been partitioned in container
        return int(value["trace_idx"] % 2 != 0)

    graph_based_distance_trace_engine = TraceBasedGraphDistanceEngine(params['engine_name'],
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


def graph_based_distance_attack(params, trace_params, output_path):
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

    graph_distance_attack_engine = GraphDistanceEngine_Attack(params['engine_name'],
                                                              selection_function,
                                                              guess_range,
                                                              time_delay=2,
                                                              dim=3,
                                                              distance_type=params['distance_type'],
                                                              num_bins=50,
                                                              solution=params['idx_of_correct_key_guess']
                                                              )
    session = Session(container,
                      engine=graph_distance_attack_engine,
                      output_method=NonIncrementalOutputMethod(
                          figure_params=params['figure_params'],
                          output_path=output_path)
                      )

    session.run(batch_size=params['batch_size'])

    del graph_distance_attack_engine
    del session


if __name__ == '__main__':
    # gt_params = graph_distance_config
    # trace_info = fixed_random_traces
    '''
    graph_based_distance
    '''
    gt_params = graph_distance_config
    trace_info = fixed_random_traces
    # json config file generation
    json_config = JSONConfig('graph_distance_v6')
    # 500k
    for m_number_of_traces in [50000, 100000, 250000, 350000]:
        for m_noise_sigma_el in [0, 0.25, 0.5, 1, 1.5, 2]:
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
        graph_based_distance(gt_params, dict_i, output_path='./results/graph_distance_v6/' + dict_i['_id'])

    '''
        graph_based_distance_trace
    '''
    gt_params = graph_distance_trace_based_config
    trace_info = fixed_random_traces
    # json config file generation
    json_config = JSONConfig('graph_distance_trace_v6')
    # 500k
    for m_number_of_traces in [50000, 100000, 250000, 350000]:
        for m_noise_sigma_el in [0, 0.25, 0.5, 1, 1.5, 2]:
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
        graph_based_distance_trace(gt_params, dict_i, output_path='./results/graph_distance_trace_v6/' + dict_i['_id'])

