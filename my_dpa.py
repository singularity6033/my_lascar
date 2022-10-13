# First the Container:
from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar.tools.aes import sbox
from Lib_SCA.lascar import DpaEngine
from Lib_SCA.lascar import Session
import matplotlib.pyplot as plt


def dpa_attack(mode, config_name, no_of_guesses=256, idx_correct_key=-1, engine_name='dpa', batch_size=2500):
    if mode == 'normal':
        container = SimulatedPowerTraceContainer(config_name)
    elif mode == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_name)
    container = SimulatedPowerTraceContainer(config_name)
    """
    Then we build the DpaEngine.
    
    If you take a look at the help for DpaEngine, you'll see that it needs 3 things to be instantiated:
    - a name for the engine ("dpa" in our case)
    - a selection function (under guess hypothesis): this function will separate the traces into two sets, depending on a hypothesis: "guess". This function will be applied on every trace values, for every possible guess.
    - a guess_range: what are the guesses you want to test?
    """

    a_byte = int(input('pls choose one byte from ' + str(container.idx_exp) + ': '))

    def selection_function(value, guess, attack_byte=a_byte, attack_time=container.attack_sample_point):
        # LSB
        return sbox[value["plaintext"][attack_byte][attack_time] ^ guess] & 1

    guess_range = range(no_of_guesses)
    dpa_engine = DpaEngine(engine_name, selection_function, guess_range)

    # We can now create a Session, register the dpa_lsb_engine, and run it.

    session = Session(container, engine=dpa_engine)
    # session.add_engine( dpa_lsb_engine) # the engine can be added after the session creation

    session.run(batch_size=batch_size)  # the session will load traces by batches of 100 traces

    """
    Now, to get the result, one solution could be to request the dpa_lsb_engine.finalize() method.
    (As most of the engines, the finalize() method returns sca results)
    
    For more option about how to manage results of sca, please follow the next step of the tutorial.
    
    """
    results = dpa_engine.finalize()

    # plotting
    plt.figure(0)
    plt.title(engine_name)
    plt.xlabel('time')
    if idx_correct_key == -1:
        plt.plot(results.T)
    else:
        for i in range(results.shape[0]):
            if i != idx_correct_key:
                plt.plot(results[i, :], color='tab:gray')
        plt.plot(results[idx_correct_key, :], color='red')
    plt.show()


if __name__ == '__main__':
    dpa_attack(mode='normal',
               config_name='normal_simulated_traces.yaml',
               no_of_guesses=256,
               idx_correct_key=0,  # the index of correct key guess
               engine_name='dpa',
               batch_size=2500)
