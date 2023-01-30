from configs.simulation_configs import normal_simulated_traces
from Lib_SCA.lascar import SimulatedPowerTraceContainer, TraceBatchContainer

container = SimulatedPowerTraceContainer(config_params=normal_simulated_traces)

# we simulate the process of generating real traces by simulated traces
real_leakages = container[:container.number_of_traces].leakages
real_values = container[:container.number_of_traces].values
idx_exp = container.idx_exp
attack_sample_point = container.attack_sample_point

#
def real_trace_generator(leakages=real_leakages,
                         values=real_values,
                         ie=idx_exp,
                         asp=attack_sample_point):
    """
    sample of real traces for attack
    """
    real_container = TraceBatchContainer(leakages, values)
    real_container.idx_exp = [ie]
    real_container.attack_sample_point = asp

    return real_container
