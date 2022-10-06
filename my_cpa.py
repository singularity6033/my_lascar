"""
In this script, we show how to perform Correlation Power Analysis with lascar to retrieve an AES keybyte.

The engine we need here is the CpaEngine.
It needs:
- a selection function (under guess hypothesis) on the sensitive value (here the output of the sbox at first round):
    This function answers the question: "Under the 'guess' hypothesis', how do I model the behavior with 'value'?
- a "guess_range" which will define what are the guess possible values

"""
from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import CpaEngine, hamming, Session, MatPlotLibOutputMethod
from Lib_SCA.lascar.tools.aes import sbox


def cpa_attack(mode, config_name, no_of_guesses=16, engine_name='cpa', batch_size=2500):
    if mode == 'normal':
        container = SimulatedPowerTraceContainer(config_name)
    elif mode == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_name)
    a_byte = int(input('pls choose one byte from ' + str(container.idx_exp) + ': '))

    def selection_function(
            value, guess, attack_byte=a_byte, attack_time=container.attack_sample_point
    ):  # selection_with_guess function must take 2 arguments: value and guess
        return hamming(sbox[value["plaintext"][attack_byte][attack_time] ^ guess])

    guess_range = range(no_of_guesses)

    cpa_engine = CpaEngine(engine_name, selection_function, guess_range)

    session = Session(
        container, engine=cpa_engine, output_method=MatPlotLibOutputMethod(cpa_engine)
    )

    session.run(batch_size=batch_size)
    results = cpa_engine.finalize()


if __name__ == '__main__':
    # mode = 'fix_random' or 'normal'
    cpa_attack(mode='fix_random',
               config_name='fixed_random_traces.yaml',
               no_of_guesses=2,
               engine_name='cpa',
               batch_size=2500)
