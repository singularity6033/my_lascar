"""
In this script, we show how to perform Correlation Power Analysis with lascar to retrieve an AES keybyte.

The engine we need here is the CpaEngine.
It needs:
- a selection function (under guess hypothesis) on the sensitive value (here the output of the sbox at first round):
    This function answers the question: "Under the 'guess' hypothesis', how do I model the behavior with 'value'?
- a "guess_range" which will define what are the guess possible values

"""

from threading import active_count

from tqdm import tqdm

from Lib_SCA.config_extractor import JSONConfig
from configs.evaluation_configs import cmi_config
from configs.simulation_configs import normal_simulated_traces
from Lib_SCA.lascar import SimulatedPowerTraceContainer
from Lib_SCA.lascar import MultipleMatrixPlotsOutputMethod
from Lib_SCA.lascar import CMI_Engine_By_Histogram, hamming, Session
from Lib_SCA.lascar.tools.aes import sbox
from real_traces_generator import real_trace_container_random


def calc_upper_bound(no_of_bytes, no_of_masking_bytes):
    return no_of_bytes * (no_of_masking_bytes + 1) * 8


def continuous_mutual_information(params, trace_params, output_path):
    container, hist_boundary = None, None
    if params['mode'] == 'normal':
        container = SimulatedPowerTraceContainer(config_params=trace_params)
        attack_byte = container.idx_exp[0]
        attack_time = container.attack_sample_point
        if not container.masking:
            up = calc_upper_bound(container.number_of_bytes, 0)
        else:
            up = calc_upper_bound(container.number_of_bytes, container.number_of_masking_bytes)
        offsets = [-3 * container.noise_sigma_el, 3 * container.noise_sigma_el]
        hist_boundary = [0 + offsets[0], up + offsets[1]]
    elif params['mode'] == 'real':
        container, t_info = real_trace_container_random(dataset_path=params['dataset_path'],
                                                        num_traces=params['num_traces'],
                                                        t_start=0,
                                                        t_end=100)
        hist_boundary = [t_info['min_leakage'], t_info['max_leakage']]

    attack_byte = params['attack_byte']

    # selection attack regions along time axis
    # container.leakage_section = params['attack_range']

    def selection_function(value, guess, ab=attack_byte):
        # LSB
        return hamming(sbox[value["plaintexts"][ab] ^ guess])

    guess_range = range(32, params['no_of_key_guesses'])

    mi_engine = CMI_Engine_By_Histogram(params['engine_name'],
                                        selection_function,
                                        guess_range,
                                        num_bins=params['num_bins'],  # 0 is 'auto' mode
                                        hist_boundary=hist_boundary,
                                        num_shuffles=params['num_shuffles'],
                                        solution=params['idx_of_correct_key_guess'])

    session = Session(container,
                      engine=mi_engine,
                      output_method=MultipleMatrixPlotsOutputMethod(
                          figure_params_along_time=params['figure_params_along_time'],
                          figure_params_along_trace=params['figure_params_along_trace'],
                          output_path=output_path,
                          display=False),
                      # output_steps=params['batch_size']
                      )
    session.run(batch_size=params['batch_size'])

    del container
    del mi_engine
    del session


if __name__ == '__main__':
    cmi_params = cmi_config
    trace_info = normal_simulated_traces
    # continuous_mutual_information(cmi_config, trace_info, output_path='./results/yzs')
    # json config file generation
    json_config = JSONConfig('cmi_test_real_v1')

    for m_number_of_traces in [1000, 5000, 10000, 50000, 100000, 250000]:
        for num_bins in [10, 20, 50]:
            cmi_params['num_traces'] = m_number_of_traces
            cmi_params['num_bins'] = num_bins
            cmi_params['_id'] = '#bins' + str(num_bins) + '_#trace_' + str(m_number_of_traces // 1000) + 'k'
            json_config.generate_config(cmi_params)

    # get json config file
    dict_list = json_config.get_config()
    for i, dict_i in tqdm(enumerate(dict_list)):
        print('[INFO] Processing #', i)
        continuous_mutual_information(dict_i, trace_info, output_path='./results/cmi_test_real_v1/' + dict_i['_id'])

    # from pathlib import Path
    #
    # #
    # p = Path("./Lib_SCA/configs/normal_simulated_traces.yaml")
    # yaml = ruamel.yaml.YAML()
    # configs = yaml.load(p)
    # configs['number_of_traces'] = 20000
    # yaml.dump(configs, p)
