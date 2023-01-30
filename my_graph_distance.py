from tqdm import tqdm

from Lib_SCA.config_extractor import JSONConfig
from configs.evaluation_configs import graph_distance_attack_config
from configs.simulation_configs import normal_simulated_traces
from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import SingleOnePlotOutputMethod, NonIncrementalOutputMethod
from Lib_SCA.lascar import Session, GraphDistanceEngine, GraphDistanceEngine_Attack
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
                                                test_type=params['test_type'],
                                                num_bins=50)

    filename = '#mask_' + str(trace_params['number_of_masking_bytes']) + '_el_' + str(trace_params['noise_sigma_el']) + \
               '_#shuffle_' + str(trace_params['shuffle']) + '_#shift_' + str(trace_params['shift']) + \
               '_#trace_' + str(trace_info['number_of_traces'] // 1000) + 'k'

    # We choose here to plot the resulting curve
    session = Session(container, engine=graph_distance_engine,
                      output_method=SingleOnePlotOutputMethod(figure_params=params['figure_params'],
                                                              output_path=output_path,
                                                              filename=filename,
                                                              display=False),
                      output_steps=params['batch_size'])

    session.run(batch_size=params['batch_size'])

    del graph_distance_engine
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
    # # json config file generation
    # json_config = JSONConfig('graph_dist_params_v1')
    # for dist in ['edit_distance', 'vertex_edge_overlap', 'vertex_edge_distance', 'lambda_dist', 'deltacon0']:
    #     for test_type in ['z-test', 't-test', 'chi-test', 'ks_2samp', 'cramervonmises_2samp']:
    #         gt_params['distance_type'] = dist
    #         gt_params['test_type'] = test_type
    #         gt_params['_id'] = dist + '+' + test_type
    #         json_config.generate_config(gt_params)
    #
    #     # get json config file
    # dict_list = json_config.get_config()
    # for i, dict_i in tqdm(enumerate(dict_list)):
    #     print('[INFO] Processing #', i)
    #     graph_based_distance(dict_i, trace_info, './results/graph_distance/' + dict_i['_id'])

    gt_params = graph_distance_attack_config
    trace_info = normal_simulated_traces
    # json config file generation
    json_config = JSONConfig('graph_dist_attack_params_v1')
    for dist in ['edit_distance']:
        gt_params['distance_type'] = dist
        for m_number_of_traces in [10000, 50000]:
            for m_noise_sigma_el in [0, 0.5]:
                for shuffle_state in [True, False]:
                    for shift_state in [True, False]:
                        for m_masking_bytes in [0, 1]:
                            trace_info['number_of_traces'] = m_number_of_traces
                            trace_info['noise_sigma_el'] = m_noise_sigma_el
                            trace_info['shuffle'] = shuffle_state
                            trace_info['shift'] = shift_state
                            trace_info['number_of_masking_bytes'] = m_masking_bytes
                            trace_info['_id'] = '#mask_' + str(m_masking_bytes) + '_el_' + str(m_noise_sigma_el) + \
                                                '_#shuffle_' + str(shuffle_state) + '_#shift_' + str(shift_state) + \
                                                '_#trace_' + str(trace_info['number_of_traces'] // 1000) + 'k'
                            trace_info['_id'] = dist + '/' + trace_info['_id']
                            json_config.generate_config(trace_info)

    # get json config file
    dict_list = json_config.get_config()
    for i, dict_i in tqdm(enumerate(dict_list)):
        print('[INFO] Processing #', i)
        graph_based_distance_attack(gt_params, dict_i, output_path='./results/graph_distance_attack_v1/' + dict_i['_id'])

    # graph_based_distance_attack(gt_params, trace_info, output_path='./results/graph_distance_attack_v1/')
