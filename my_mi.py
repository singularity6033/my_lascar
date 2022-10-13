"""
In this script, we show how to perform Correlation Power Analysis with lascar to retrieve an AES keybyte.

The engine we need here is the CpaEngine.
It needs:
- a selection function (under guess hypothesis) on the sensitive value (here the output of the sbox at first round):
    This function answers the question: "Under the 'guess' hypothesis', how do I model the behavior with 'value'?
- a "guess_range" which will define what are the guess possible values

"""
from matplotlib import pyplot as plt

from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import CMI_Engine, hamming, Session, MatPlotLibOutputMethod
from Lib_SCA.lascar.tools.aes import sbox


def cmi(mode, config_name, no_of_guesses=256, idx_correct_key=-1, contain_test=True, engine_name='cmi', batch_size=2500):
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

    mi_engine = CMI_Engine(engine_name, selection_function, guess_range, contain_test=contain_test)

    session = Session(container, engine=mi_engine)

    session.run(batch_size=batch_size)
    if contain_test:
        real_mi, test_score = mi_engine.finalize()
    else:
        real_mi = mi_engine.finalize()

    # plotting
    plt.figure(0)
    plt.title(engine_name)
    plt.xlabel('time')
    plt.ylabel('continuous mutual information')
    if idx_correct_key == -1:
        plt.plot(real_mi.T)
    else:
        for i in range(real_mi.shape[0]):
            if i != idx_correct_key:
                plt.plot(real_mi[i, :], color='tab:gray')
        plt.plot(real_mi[idx_correct_key, :], color='red')
        plt.show()
        if contain_test:
            # test score
            plt.figure(1)
            plt.title(engine_name)
            plt.xlabel('time')
            plt.ylabel('test score (%)')
            for i in range(test_score.shape[0]):
                if i != idx_correct_key:
                    plt.plot(test_score[i, :], color='tab:gray')
            plt.plot(test_score[idx_correct_key, :], color='red')
            plt.show()


if __name__ == '__main__':
    # mode = 'fix_random' or 'normal'
    cmi(mode='normal',
        config_name='normal_simulated_traces.yaml',
        no_of_guesses=4,
        idx_correct_key=0,  # the index of correct key guess
        contain_test=False,
        engine_name='cmi',
        batch_size=3000)
