from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import CpaEngine, hamming, Session, MatPlotLibOutputMethod
from Lib_SCA.lascar.tools.aes import sbox

container = SimulatedPowerTraceContainer('normal_simulated_traces.yaml')

def selection_function(value, guess):
    result = hamming(sbox[value['plaintext'][0][0]^guess])
    return result

cpa_engine = CpaEngine('cpa',selection_function,range(256))

cpa_session = Session(container, engine=cpa_engine, output_method=MatPlotLibOutputMethod(cpa_engine))

cpa_session.run(batch_size=100)


