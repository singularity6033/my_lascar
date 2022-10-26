"""
In this script, we show how to perform Correlation Power Analysis with lascar to retrieve an AES keybyte.

The engine we need here is the CpaEngine.
It needs:
- a selection function (under guess hypothesis) on the sensitive value (here the output of the sbox at first round):
    This function answers the question: "Under the 'guess' hypothesis', how do I model the behavior with 'value'?
- a "guess_range" which will define what are the guess possible values

"""
import numpy as np
from matplotlib import pyplot as plt

from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import CMI_Engine_By_Histogram, hamming, Session, MatPlotLibOutputMethod
from Lib_SCA.lascar.tools.aes import sbox


def cmi(mode,
        config_name,
        no_of_guesses=256,
        idx_correct_key=-1,
        engine_name='cmi',
        num_bins=5,
        num_shuffles=100,
        batch_size=2500):

    if mode == 'normal':
        container = SimulatedPowerTraceContainer(config_name,
                                                 # seed=1
                                                 )
    elif mode == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_name)
    a_byte = int(input('pls choose one byte from ' + str(container.idx_exp) + ': '))

    def selection_function(
            value, guess, attack_byte=a_byte, attack_time=container.attack_sample_point
    ):  # selection_with_guess function must take 2 arguments: value and guess
        return hamming(sbox[value["plaintext"][attack_byte][attack_time] ^ guess])

    guess_range = range(no_of_guesses)

    mi_engine = CMI_Engine_By_Histogram(engine_name,
                                        selection_function,
                                        guess_range,
                                        num_bins=num_bins,
                                        # hist_mode='merge',
                                        num_shuffles=num_shuffles)

    session = Session(container, engine=mi_engine)

    session.run(batch_size=batch_size)
    mi, pv = mi_engine.finalize()

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
        plt.show()
        plt.figure(1)
        plt.title(engine_name + '+pv')
        for i in range(pv.shape[0]):
            if i != idx_correct_key:
                plt.plot(pv[i, :], color='tab:gray')
        plt.plot(pv[idx_correct_key, :], color='red')
        plt.show()


if __name__ == '__main__':
    # mode = 'fix_random' or 'normal'
    cmi(mode='normal',
        config_name='normal_simulated_traces.yaml',
        no_of_guesses=3,
        idx_correct_key=0,  # the index of correct key guess
        engine_name='cmi',
        num_bins=10,
        num_shuffles=100,
        batch_size=3000)
