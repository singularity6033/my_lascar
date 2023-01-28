"""
In this script, we show how to perform Correlation Power Analysis with lascar to retrieve an AES keybyte.

The engine we need here is the CpaEngine.
It needs:
- a selection function (under guess hypothesis) on the sensitive value (here the output of the sbox at first round):
    This function answers the question: "Under the 'guess' hypothesis', how do I model the behavior with 'value'?
- a "guess_range" which will define what are the guess possible values

"""

from threading import active_count
from configs.evaluation_configs import cmi_config
from configs.simulation_configs import normal_simulated_traces
from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import MultipleMatrixPlotsOutputMethod
from Lib_SCA.lascar import CMI_Engine_By_Histogram, hamming, Session
from Lib_SCA.lascar.tools.aes import sbox


def continuous_mutual_information(params, trace_params):
    params = params
    container = None
    if params['mode'] == 'normal':
        container = SimulatedPowerTraceContainer(config_params=trace_params)
    elif params['mode'] == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_params=trace_params)
    elif params['mode'] == 'real':
        pass

    a_byte = container.idx_exp[0]
    attack_time = container.attack_sample_point

    # selection attack regions along time axis
    # container.leakage_section = params['attack_range']

    def selection_function(
            value, guess, ab=a_byte, at=attack_time
    ):  # selection_with_guess function must take 2 arguments: value and guess
        return hamming(sbox[value["plaintext"][ab][at] ^ guess])

    guess_range = range(params['no_of_key_guesses'])

    def calc_best_num_of_hist_bins(no_of_bytes, no_of_masking_bytes):
        return no_of_bytes * (no_of_masking_bytes + 1) * 8 + 1

    if not container.masking:
        num_bins = calc_best_num_of_hist_bins(container.number_of_bytes, 0)
    else:
        num_bins = calc_best_num_of_hist_bins(container.number_of_bytes, container.number_of_masking_bytes)  # or 0 ('auto')
    hist_boundary = [0, num_bins-1]  # or None

    mi_engine = CMI_Engine_By_Histogram(params['engine_name'],
                                        selection_function,
                                        guess_range,
                                        num_bins=num_bins,
                                        hist_boundary=hist_boundary,
                                        num_shuffles=params['num_shuffles'],
                                        solution=params['idx_of_correct_key_guess'])
    # trace_params['_id']
    output_path = './plots/cmi_results_test/' + 'yzs'
    session = Session(container,
                      engine=mi_engine,
                      output_method=MultipleMatrixPlotsOutputMethod(figure_params_along_time=params['figure_params_along_time'],
                                                                    figure_params_along_trace=params['figure_params_along_trace'],
                                                                    output_path=output_path,
                                                                    display=False),
                      # output_steps=params['batch_size']
                      )
    session.run(batch_size=params['batch_size'])

    del container
    del mi_engine
    del session
    print(active_count())


if __name__ == '__main__':
    cmi_params = cmi_config
    trace_info = normal_simulated_traces
    # # json config file generation
    # json_config = JSONConfig('cmi_test_1')
    # tracemalloc.start()
    # # 10k, 50k, 200k, 500k, 1000k
    # for m_number_of_traces in [500000, 10000000]:
    #     for m_number_of_bytes in range(1, 17):
    #         for m_noise_sigma_el in [0, 0.25, 0.5]:
    #             for m_num_of_masking_bytes in [0, 1, 2]:
    #                 m_idx_switching_noise_bytes = [i + 1 for i in range(m_number_of_bytes - 1)]
    #                 trace_info['number_of_traces'] = m_number_of_traces
    #                 trace_info['number_of_bytes'] = m_number_of_bytes
    #                 trace_info['idx_switching_noise_bytes'] = m_idx_switching_noise_bytes
    #                 trace_info["number_of_masking_bytes"] = m_num_of_masking_bytes
    #                 trace_info['noise_sigma_el'] = m_noise_sigma_el
    #                 trace_info['_id'] = '#mask_' + str(trace_info["number_of_masking_bytes"]) + \
    #                                     '_el_' + str(trace_info['noise_sigma_el']) + \
    #                                     '_#switch_' + str(trace_info['number_of_bytes'] - 1) + \
    #                                     '_#trace_' + str(trace_info['number_of_traces'] // 1000) + 'k'
    #                 json_config.generate_config(trace_info)
    #
    # # get json config file
    # dict_list = json_config.get_config()
    # for i, dict_i in tqdm(enumerate(dict_list[1:])):
    #     print('[INFO] Processing #', i)
    #     continuous_mutual_information(cmi_params, dict_i)
    continuous_mutual_information(cmi_params, trace_info)

    # from pathlib import Path
    #
    # #
    # p = Path("./Lib_SCA/configs/normal_simulated_traces.yaml")
    # yaml = ruamel.yaml.YAML()
    # configs = yaml.load(p)
    # configs['number_of_traces'] = 20000
    # yaml.dump(configs, p)
