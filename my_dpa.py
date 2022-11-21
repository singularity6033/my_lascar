from Lib_SCA.config_extractor import TraceConfig
from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer, \
    Single_Result_OutputMethod
from Lib_SCA.lascar.tools.aes import sbox
from Lib_SCA.lascar import DpaEngine
from Lib_SCA.lascar import Session
import matplotlib.pyplot as plt

from real_traces_generator import real_trace_generator


def dpa_attack(config_name):
    dpa_params = TraceConfig().get_config(config_name)
    if dpa_params['mode'] == 'normal':
        container_params = TraceConfig().get_config('normal_simulated_traces.yaml')
        container = SimulatedPowerTraceContainer(config_params=container_params)
    elif dpa_params['mode'] == 'fix_random':
        container_params = TraceConfig().get_config('fixed_random_traces.yaml')
        container = SimulatedPowerTraceFixedRandomContainer(config_params=container_params)
    elif dpa_params['mode'] == 'real':
        container, idx_exp, attack_time = real_trace_generator()
        a_byte = int(input('pls choose one byte from ' + str(idx_exp) + ': '))
    if not dpa_params['mode'] == 'real':
        a_byte = int(input('pls choose one byte from ' + str(container.idx_exp) + ': '))
        attack_time = container.attack_sample_point

    # selection attack regions along time axis
    # container.leakage_section = eval(dpa_params['attack_range'])
    """
    Then we build the DpaEngine.
    
    If you take a look at the help for DpaEngine, you'll see that it needs 3 things to be instantiated:
    - a name for the engine ("dpa" in our case)
    - a selection function (under guess hypothesis): this function will separate the traces into two sets, depending on a hypothesis: "guess". This function will be applied on every trace values, for every possible guess.
    - a guess_range: what are the guesses you want to test?
    """

    a_byte = int(input('pls choose one byte from ' + str(container.idx_exp) + ': '))

    def selection_function(value, guess, attack_byte=a_byte, attack_time=attack_time):
        # LSB
        return sbox[value["plaintext"][attack_byte][attack_time] ^ guess] & 1

    guess_range = range(dpa_params['no_of_key_guesses'])

    dpa_engine = DpaEngine(dpa_params['engine_name'],
                           selection_function,
                           guess_range,
                           solution=dpa_params['idx_of_correct_key_guess'])

    session = Session(container,
                      engine=dpa_engine,
                      output_method=Single_Result_OutputMethod(figure_params=dpa_params['figure_params'],
                                                               output_path='./plots/dpa.png'))

    session.run(batch_size=dpa_params['batch_size'])


if __name__ == '__main__':
    dpa_attack(config_name='dpa_attack.yaml')
