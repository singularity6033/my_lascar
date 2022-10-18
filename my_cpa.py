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
from Lib_SCA.lascar import CpaEngine, hamming, Session, MatPlotLibOutputMethod, numerical_success_rate
from Lib_SCA.lascar.tools.aes import sbox
from real_traces_generator import real_trace_generator


def cpa_attack(config_name):
    params = TraceConfig().get_config(config_name)
    if params['mode'] == 'normal':
        container = SimulatedPowerTraceContainer('normal_simulated_traces.yaml')
    elif params['mode'] == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer('fixed_random_traces.yaml')
    elif params['mode'] == 'real':
        container, idx_exp, attack_time = real_trace_generator()
        a_byte = int(input('pls choose one byte from ' + str(idx_exp) + ': '))
    if not params['mode'] == 'real':
        a_byte = int(input('pls choose one byte from ' + str(container.idx_exp) + ': '))
        attack_time = container.attack_sample_point
    container.leakage_section = eval(params['attack_range'])

    def selection_function(
            value, guess, attack_byte=a_byte, at=attack_time
    ):  # selection_with_guess function must take 2 arguments: value and guess
        return hamming(sbox[value["plaintext"][attack_byte][at] ^ guess])

    guess_range = range(params['no_of_key_guesses'])

    cpa_engine = CpaEngine(params['engine_name'], selection_function, guess_range)

    session = Session(container, engine=cpa_engine)

    session.run(batch_size=params['batch_size'])
    results = cpa_engine.finalize()

    # plotting
    plt.figure(0)
    plt.title(params['engine_name'])
    plt.xlabel('time')
    if params['idx_of_correct_key_guess'] == -1:
        plt.plot(results.T)
    else:
        for i in range(results.shape[0]):
            if i != params['idx_of_correct_key_guess']:
                plt.plot(results[i, :], color='tab:gray')
        plt.plot(results[params['idx_of_correct_key_guess'], :], color='red')
    plt.show()
    # print(numerical_success_rate(distinguish_vector=results, correct_key=0, order=1).eval())


if __name__ == '__main__':
    cpa_attack(config_name='cpa_attack.yaml')
