"""
In this script, we show how to perform Correlation Power Analysis with lascar to retrieve an AES keybyte.

The engine we need here is the CpaEngine.
It needs:
- a selection function (under guess hypothesis) on the sensitive value (here the output of the sbox at first round):
    This function answers the question: "Under the 'guess' hypothesis', how do I model the behavior with 'value'?
- a "guess_range" which will define what are the guess possible values

"""
import os
from tqdm import tqdm
from Lib_SCA.config_extractor import JSONConfig
from configs.attack_configs import cpa_config
from configs.simulation_configs import normal_simulated_traces
from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import CpaEngine, hamming, Session, SingleMatrixPlotOutputMethod
from Lib_SCA.lascar.tools.aes import sbox


# from real_traces_generator import real_trace_generator


def cpa_attack(params, trace_params, output_path):
    container = None
    if params['mode'] == 'normal':
        container = SimulatedPowerTraceContainer(config_params=trace_params)
    elif params['mode'] == 'real':
        pass

    # attack bytes and attack time sample point
    a_byte = container.idx_exp[0]
    attack_time = container.attack_sample_point

    # selection attack regions along the time axis
    # container.leakage_section = params['attack_range']

    def selection_function(
            value, guess, attack_byte=a_byte, at=attack_time
    ):  # selection_with_guess function must take 2 arguments: value and guess
        return hamming(sbox[value["plaintext"][attack_byte][at] ^ guess])

    guess_range = range(params['no_of_key_guesses'])

    cpa_engine = CpaEngine(params['engine_name'],
                           selection_function,
                           guess_range,
                           solution=params['idx_of_correct_key_guess'])

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    session = Session(container,
                      engine=cpa_engine,
                      output_method=SingleMatrixPlotOutputMethod(
                          figure_params_along_time=params['figure_params_along_time'],
                          figure_params_along_trace=params['figure_params_along_trace'],
                          output_path=output_path,
                          filename='cpa',
                          contain_raw_file=True),
                      output_steps=200)

    session.run(batch_size=params['batch_size'])
    # results = cpa_engine.finalize()
    # print(numerical_success_rate(distinguish_vector=results, correct_key=0, order=1).eval())


if __name__ == '__main__':
    # for the single trail (just use parameters once)
    cpa_attack(cpa_config, normal_simulated_traces, output_path='./results/cpa')

    # # for the multiple trails (parameters sweeping)
    # # initial parameters for CPA and simulated traces
    # gt_params = cpa_config
    # trace_info = normal_simulated_traces
    #
    # # create .json file to save all parameters setting
    # json_config = JSONConfig('graph_direct_test_attack_v1')\
    #
    # for m_number_of_traces in [5000, 10000, 50000]:
    #     for m_noise_sigma_el in [0, 0.5]:
    #         for shuffle_state in [True, False]:
    #             for shift_state in [True, False]:
    #                 trace_info['number_of_traces'] = m_number_of_traces
    #                 trace_info['noise_sigma_el'] = m_noise_sigma_el
    #                 trace_info['shuffle'] = shuffle_state
    #                 trace_info['shift'] = shift_state
    #                 # create '_id' to identify each record
    #                 trace_info['_id'] = '_el_' + str(m_noise_sigma_el) + '_#shift_' + str(shift_state) + \
    #                                     '_#trace_' + str(trace_info['number_of_traces'] // 1000) + 'k'
    #                 json_config.generate_config(trace_info)
    #
    # # get .json file just created and read all parameters setting
    # dict_list = json_config.get_config()
    # for i, dict_i in tqdm(enumerate(dict_list)):
    #     print('[INFO] Processing #', i)
    #     cpa_attack(gt_params, dict_i, './results/cpa/' + dict_i['_id'])

