from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer, TraceBatchContainer
from Lib_SCA.lascar import CpaEngine, hamming, Session, MatPlotLibOutputMethod
from Lib_SCA.lascar.tools.aes import sbox

container = SimulatedPowerTraceContainer('normal_simulated_traces.yaml')
# we simulate the process of generating real traces by simulated traces
real_leakages = container[:container.number_of_traces].leakages
real_values = container[:container.number_of_traces].values

real_container = TraceBatchContainer(real_leakages, real_values)
real_container.leakage_section = range(4)


def selection_function(value, guess):
    result = hamming(sbox[value['plaintext'][0][0] ^ guess])
    return result


cpa_engine = CpaEngine('cpa', selection_function, range(256), jit=False)

cpa_session = Session(real_container, engine=cpa_engine, output_method=MatPlotLibOutputMethod(cpa_engine))
#PLI
cpa_session.run(batch_size=2500)
