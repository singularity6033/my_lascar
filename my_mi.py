"""
In this script, we show how to perform Correlation Power Analysis with lascar to retrieve an AES keybyte.

The engine we need here is the CpaEngine.
It needs:
- a selection function (under guess hypothesis) on the sensitive value (here the output of the sbox at first round):
    This function answers the question: "Under the 'guess' hypothesis', how do I model the behavior with 'value'?
- a "guess_range" which will define what are the guess possible values

"""
from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import MiEngine, Session


def mi(mode, config_name, engine_name='cpa', batch_size=2500):
    if mode == 'normal':
        container = SimulatedPowerTraceContainer(config_name)
    elif mode == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_name)

    mi_engine = MiEngine(engine_name)

    session = Session(container, engine=mi_engine)

    session.run(batch_size=batch_size)
    results = mi_engine.finalize()
    print(results)


if __name__ == '__main__':
    # mode = 'fix_random' or 'normal'
    mi(mode='normal',
       config_name='normal_simulated_traces.yaml',
       engine_name='mi',
       batch_size=2500)
