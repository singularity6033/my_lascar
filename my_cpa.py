"""
In this script, we show how to perform Correlation Power Analysis with lascar to retrieve an AES keybyte.

The engine we need here is the CpaEngine.
It needs:
- a selection function (under guess hypothesis) on the sensitive value (here the output of the sbox at first round):
    This function answers the question: "Under the 'guess' hypothesis', how do I model the behavior with 'value'?
- a "guess_range" which will define what are the guess possible values

"""
from Lib_SCA.lascar import SimulatedPowerTraceContainer, CpaEngine, hamming, Session, MatPlotLibOutputMethod
from Lib_SCA.lascar.tools.aes import sbox

container = SimulatedPowerTraceContainer('normal_simulated_traces.yaml')


def selection_function(
        value, guess
):  # selection_with_guess function must take 2 arguments: value and guess
    """
    What would the hamming weight of the output of the 3rd sbox be if the key was equal to 'guess' ?
    """
    return hamming(sbox[value["plaintext"][0][0] ^ guess])


guess_range = range(
    16
)  # the guess values: here we make hypothesis on a key byte, hence range(256)

cpa_engine = CpaEngine("cpa_plaintext_3", selection_function, guess_range)

session = Session(
    container, engine=cpa_engine, output_method=MatPlotLibOutputMethod(cpa_engine)
)

session.run(batch_size=200)
