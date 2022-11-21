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
from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import CMI_Engine_By_Histogram, hamming, Session, MatPlotLibOutputMethod
from Lib_SCA.lascar.tools.aes import sbox
import ruamel.yaml


def calc_best_num_of_hist_bins(no_of_bytes, no_of_masking_bytes):
    return no_of_bytes * (no_of_masking_bytes + 1) * 8 + 1


params = TraceConfig().get_config('normal_simulated_traces.yaml')


def cmi(mode,
        config_name,
        no_of_guesses=256,
        idx_correct_key=-1,
        engine_name='cmi',
        num_bins=5,
        hist_boundary=None,
        num_shuffles=100,
        batch_size=2500):
    if mode == 'normal':
        container = SimulatedPowerTraceContainer(config_name)
    elif mode == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_name)

    # a_byte = int(input('pls choose one byte from ' + str(container.idx_exp) + ': '))
    a_byte = 0

    def selection_function(
            value, guess, attack_byte=a_byte, attack_time=container.attack_sample_point
    ):  # selection_with_guess function must take 2 arguments: value and guess
        return hamming(sbox[value["plaintext"][attack_byte][attack_time] ^ guess])

    guess_range = range(no_of_guesses)

    mi_engine = CMI_Engine_By_Histogram(engine_name,
                                        selection_function,
                                        guess_range,
                                        num_bins=num_bins,
                                        hist_boundary=hist_boundary,
                                        num_shuffles=num_shuffles)

    session = Session(container, engine=mi_engine)

    session.run(batch_size=batch_size)
    mi, pv = mi_engine.results

    # plotting
    if idx_correct_key == -1:
        plt.figure(0)
        plt.title(engine_name + '+mi')
        plt.plot(mi.T)
        plt.show()
        plt.show()
        plt.figure(1)
        plt.title(engine_name + '+pv')
        plt.plot(pv.T)
        plt.show()
    else:
        plt.figure(0)
        plt.title(engine_name + '+mi')
        for i in range(mi.shape[0]):
            if i != idx_correct_key:
                plt.plot(mi[i, :], color='tab:gray')
        plt.plot(mi[idx_correct_key, :], color='red')
        # plt.show()
        plt.savefig('./plots/#mask_' + str(params['num_of_masking_bytes']) + '_el_' + str(
            params['noise_sigma_el']) + '_#switch_'
                    + str(params['number_of_bytes'] - 1) + '_mi_' + str(params['number_of_traces'] // 1000) + '.png')
        plt.figure(1)
        plt.title(engine_name + '+pv')
        for i in range(pv.shape[0]):
            if i != idx_correct_key:
                plt.plot(pv[i, :], color='tab:gray')
        plt.plot(pv[idx_correct_key, :], color='red')
        plt.show()
        plt.savefig('./plots/#mask_' + str(params['num_of_masking_bytes']) + '_el_' + str(
            params['noise_sigma_el']) + '_#switch_'
                    + str(params['number_of_bytes'] - 1) + '_pv_' + str(params['number_of_traces'] // 1000) + '.png')


if __name__ == '__main__':
    # mode = 'fix_random' or 'normal'
    # cmi(mode='normal',
    #     config_name='normal_simulated_traces.yaml',
    #     no_of_guesses=50,
    #     idx_correct_key=0,  # the index of correct key guess
    #     engine_name='cmi',
    #     num_bins=calc_best_num_of_hist_bins(params['number_of_bytes'], params['num_of_masking_bytes']),
    #     # num_bins=0,
    #     hist_boundary=[0, calc_best_num_of_hist_bins(params['number_of_bytes'], params['num_of_masking_bytes'])],
    #     # hist_boundary=None,
    #     num_shuffles=25,
    #     batch_size=1000000)

    from pathlib import Path
    #
    p = Path("./Lib_SCA/configs/normal_simulated_traces.yaml")
    yaml = ruamel.yaml.YAML()
    configs = yaml.load(p)
    configs['number_of_traces'] = 20000
    yaml.dump(configs, p)
