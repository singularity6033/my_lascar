"""
In this script, we show how to perform Correlation Power Analysis with lascar to retrieve an AES keybyte.

The engine we need here is the CpaEngine.
It needs:
- a selection function (under guess hypothesis) on the sensitive value (here the output of the sbox at first round):
    This function answers the question: "Under the 'guess' hypothesis', how do I model the behavior with 'value'?
- a "guess_range" which will define what are the guess possible values

"""
import sys

import numpy as np
from matplotlib import pyplot as plt
from Lib_SCA.config_extractor import TraceConfig
from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer, \
    Multiple_Results_OutputMethod
from Lib_SCA.lascar import CMI_Engine_By_Histogram, hamming, Session, MatPlotLibOutputMethod
from Lib_SCA.lascar.tools.aes import sbox
import ruamel.yaml


def calc_best_num_of_hist_bins(no_of_bytes, no_of_masking_bytes):
    return no_of_bytes * (no_of_masking_bytes + 1) * 8 + 1


params = TraceConfig().get_config('normal_simulated_traces.yaml')


def cmi(config_name, **kwargs):
    cmi_params = TraceConfig().get_config(config_name)
    if cmi_params['mode'] == 'normal':
        container_params = TraceConfig().get_config('normal_simulated_traces.yaml')
        if cmi_params['scan']:
            container_params['number_of_traces'] = kwargs['m_number_of_traces']
            container_params['number_of_bytes'] = kwargs['m_number_of_bytes']
            container_params['idx_switching_noise_bytes'] = kwargs['m_idx_switching_noise_bytes']
            container_params['noise_sigma_el'] = kwargs['m_noise_sigma_el']
            container_params['num_of_masking_bytes'] = kwargs['m_num_of_masking_bytes']
        container = SimulatedPowerTraceContainer(config_params=container_params)
    elif cmi_params['mode'] == 'fix_random':
        container_params = TraceConfig().get_config('fixed_random_traces.yaml')
        if cmi_params['scan']:
            container_params['number_of_traces'] = kwargs['m_number_of_traces']
            container_params['number_of_bytes'] = kwargs['m_number_of_bytes']
            container_params['idx_switching_noise_bytes'] = kwargs['m_idx_switching_noise_bytes']
            container_params['noise_sigma_el'] = kwargs['m_noise_sigma_el']
            container_params['num_of_masking_bytes'] = kwargs['m_num_of_masking_bytes']
        container = SimulatedPowerTraceFixedRandomContainer(config_params=container_params)
    elif cmi_params['mode'] == 'real':
        pass
    if not cmi_params['mode'] == 'real':
        a_byte = 0
        attack_time = container.attack_sample_point

    # selection attack regions along time axis
    # container.leakage_section = eval(cmi_params['attack_range'])

    def selection_function(
            value, guess, attack_byte=a_byte, attack_time=attack_time
    ):  # selection_with_guess function must take 2 arguments: value and guess
        return hamming(sbox[value["plaintext"][attack_byte][attack_time] ^ guess])

    guess_range = range(cmi_params['no_of_guesses'])

    num_bins = calc_best_num_of_hist_bins(container.number_of_bytes, container.number_of_masking_bytes)  # or 0 ('auto')
    hist_boundary = [0, num_bins]  # or None

    mi_engine = CMI_Engine_By_Histogram(cmi_params['engine_name'],
                                        selection_function,
                                        guess_range,
                                        num_bins=num_bins,
                                        hist_boundary=hist_boundary,
                                        num_shuffles=cmi_params['num_shuffles'],
                                        solution=cmi_params['idx_correct_key'])

    output_path = './plots/cmi/#mask_' + str(container.number_of_masking_bytes) + '_el_' \
                  + str(container.noise_sigma_el) + '_#switch_' + str(container.number_of_bytes - 1) \
                  + '_nt_' + str(container.number_of_traces // 1000) + 'k.png'
    session = Session(container,
                      engine=mi_engine,
                      output_method=Multiple_Results_OutputMethod(figure_params=cmi_params['figure_params'],
                                                                  output_path='./plots/cpa.png'))
    session.run(batch_size=cmi_params['batch_size'])


if __name__ == '__main__':
    cmi(config_name='cmi.yaml',
        m_number_of_traces=1000,
        m_number_of_bytes=5,
        m_idx_switching_noise_bytes=[i+1 for i in range(4)],
        m_noise_sigma_el=0.2,
        m_num_of_masking_bytes=2)

    # from pathlib import Path
    #
    # #
    # p = Path("./Lib_SCA/configs/normal_simulated_traces.yaml")
    # yaml = ruamel.yaml.YAML()
    # configs = yaml.load(p)
    # configs['number_of_traces'] = 20000
    # yaml.dump(configs, p)
