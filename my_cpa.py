"""
In this script, we show how to perform Correlation Power Analysis with lascar to retrieve an AES keybyte.

The engine we need here is the CpaEngine.
It needs:
- a selection function (under guess hypothesis) on the sensitive value (here the output of the sbox at first round):
    This function answers the question: "Under the 'guess' hypothesis', how do I model the behavior with 'value'?
- a "guess_range" which will define what are the guess possible values

"""
from matplotlib import pyplot as plt

from Lib_SCA.config_extractor import TraceConfig
from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import CpaEngine, hamming, Session, Single_Result_OutputMethod, numerical_success_rate
from Lib_SCA.lascar.tools.aes import sbox
from real_traces_generator import real_trace_generator


def cpa_attack(config_name):
    cpa_params = TraceConfig().get_config(config_name)
    if cpa_params['mode'] == 'normal':
        container_params = TraceConfig().get_config('normal_simulated_traces.yaml')
        container = SimulatedPowerTraceContainer(config_params=container_params)
    elif cpa_params['mode'] == 'fix_random':
        container_params = TraceConfig().get_config('fixed_random_traces.yaml')
        container = SimulatedPowerTraceFixedRandomContainer(config_params=container_params)
    elif cpa_params['mode'] == 'real':
        container, idx_exp, attack_time = real_trace_generator()
        a_byte = int(input('pls choose one byte from ' + str(idx_exp) + ': '))
    if not cpa_params['mode'] == 'real':
        a_byte = int(input('pls choose one byte from ' + str(container.idx_exp) + ': '))
        attack_time = container.attack_sample_point

    # selection attack regions along time axis
    # container.leakage_section = eval(cpa_params['attack_range'])

    def selection_function(
            value, guess, attack_byte=a_byte, at=attack_time
    ):  # selection_with_guess function must take 2 arguments: value and guess
        return hamming(sbox[value["plaintext"][attack_byte][at] ^ guess])

    guess_range = range(cpa_params['no_of_key_guesses'])

    cpa_engine = CpaEngine(cpa_params['engine_name'],
                           selection_function,
                           guess_range,
                           solution=cpa_params['idx_of_correct_key_guess'])

    session = Session(container,
                      engine=cpa_engine,
                      output_method=Single_Result_OutputMethod(figure_params=cpa_params['figure_params'],
                                                               output_path='./plots/cpa.png'))

    session.run(batch_size=cpa_params['batch_size'])
    # results = cpa_engine.finalize()
    # print(numerical_success_rate(distinguish_vector=results, correct_key=0, order=1).eval())


if __name__ == '__main__':
    cpa_attack(config_name='cpa_attack.yaml')
