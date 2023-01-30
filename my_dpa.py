import os

from configs.attack_configs import dpa_config
from configs.simulation_configs import normal_simulated_traces
from Lib_SCA.lascar import SimulatedPowerTraceContainer
from Lib_SCA.lascar import SingleMatrixPlotOutputMethod
from Lib_SCA.lascar.tools.aes import sbox
from Lib_SCA.lascar import DpaEngine
from Lib_SCA.lascar import Session


# from real_traces_generator import real_trace_generator


def dpa_attack(params, trace_params, output_path):
    container = None
    if params['mode'] == 'normal':
        container = SimulatedPowerTraceContainer(config_params=trace_params)
    elif params['mode'] == 'real':
        pass
        # container = real_trace_generator()

    attack_byte = container.idx_exp[0]
    attack_time = container.attack_sample_point

    # selection attack regions along time axis
    # container.leakage_section = params['attack_range']
    """
    Then we build the DpaEngine.
    
    If you take a look at the help for DpaEngine, you'll see that it needs 3 things to be instantiated:
    - a name for the engine ("dpa" in our case)
    - a selection function (under guess hypothesis): this function will separate the traces into two sets, depending on a hypothesis: "guess". This function will be applied on every trace values, for every possible guess.
    - a guess_range: what are the guesses you want to test?
    """

    def selection_function(value, guess, ab=attack_byte, at=attack_time):
        # LSB
        return sbox[value["plaintext"][ab][at] ^ guess] & 1

    guess_range = range(params['no_of_key_guesses'])

    dpa_engine = DpaEngine(params['engine_name'],
                           selection_function,
                           guess_range,
                           solution=params['idx_of_correct_key_guess'])

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    session = Session(container,
                      engine=dpa_engine,
                      output_method=SingleMatrixPlotOutputMethod(
                          figure_params_along_time=params['figure_params_along_time'],
                          figure_params_along_trace=params['figure_params_along_trace'],
                          output_path=output_path,
                          filename='dpa',
                          contain_raw_file=True),
                      output_steps=params['batch_size'])

    session.run(batch_size=params['batch_size'])


if __name__ == '__main__':
    dpa_attack(dpa_config, normal_simulated_traces, output_path='./results/dpa')
