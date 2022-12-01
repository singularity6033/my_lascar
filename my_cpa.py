"""
In this script, we show how to perform Correlation Power Analysis with lascar to retrieve an AES keybyte.

The engine we need here is the CpaEngine.
It needs:
- a selection function (under guess hypothesis) on the sensitive value (here the output of the sbox at first round):
    This function answers the question: "Under the 'guess' hypothesis', how do I model the behavior with 'value'?
- a "guess_range" which will define what are the guess possible values

"""
from matplotlib import pyplot as plt

from Lib_SCA.config_extractor import YAMLConfig, JSONConfig
from Lib_SCA.configs.attack_configs import cpa_config
from Lib_SCA.configs.simulation_configs import normal_simulated_traces, fixed_random_traces
from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import CpaEngine, hamming, Session, Single_Result_OutputMethod, numerical_success_rate
from Lib_SCA.lascar.tools.aes import sbox
from real_traces_generator import real_trace_generator


def cpa_attack(params, trace_params):
    container = None
    if params['mode'] == 'normal':
        container = SimulatedPowerTraceContainer(config_params=trace_params)
    elif params['mode'] == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_params=trace_params)
    elif params['mode'] == 'real':
        container = real_trace_generator()

    a_byte = container.idx_exp[0]
    attack_time = container.attack_sample_point

    # selection attack regions along time axis
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

    session = Session(container,
                      engine=cpa_engine,
                      output_method=Single_Result_OutputMethod(figure_params=params['figure_params'],
                                                               output_path='./plots/cpa.png'))

    session.run(batch_size=params['batch_size'])
    # results = cpa_engine.finalize()
    # print(numerical_success_rate(distinguish_vector=results, correct_key=0, order=1).eval())


if __name__ == '__main__':
    cpa_attack(cpa_config, normal_simulated_traces)
