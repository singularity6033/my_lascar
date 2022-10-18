from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer, TraceBatchContainer

container = SimulatedPowerTraceContainer('normal_simulated_traces.yaml')

# we simulate the process of generating real traces by simulated traces
real_leakages = container[:container.number_of_traces].leakages
real_values = container[:container.number_of_traces].values
idx_exp = container.idx_exp
attack_sample_point = container.attack_sample_point


def real_trace_generator(leakages=real_leakages, values=real_values, idx_exp=idx_exp,
                         attack_sample_point=attack_sample_point):
    real_container = TraceBatchContainer(leakages, values)

    return real_container, idx_exp, attack_sample_point
