"""
In this script, we show how to perform side-channel characterisation using Welch's T-test to study the behaviour of an Aes Sbox

The characterisation is made with the TTestEngine
Its constructor needs a partition function, which will separate leakages into two classes.

"""
from configs.evaluation_configs import chi2_test_config
from configs.simulation_configs import fixed_random_traces
from Lib_SCA.lascar import SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import SingleVectorPlotOutputMethod
from Lib_SCA.lascar import Session, Chi2TestEngine


# from real_traces_generator import real_trace_generator


def chi2_test(params, trace_params):
    container = None
    if params['mode'] == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_params=trace_params)

    def partition_function(value):
        # partition_function must take 1 argument: the value returned by the container at each trace
        # fix and random sets have already been partitioned in container
        return int(value["trace_idx"] % 2 == 0)

    def calc_best_num_of_hist_bins(no_of_bytes, no_of_masking_bytes):
        return no_of_bytes * (no_of_masking_bytes + 1) * 8 + 1

    if not container.masking:
        num_bins = calc_best_num_of_hist_bins(container.number_of_bytes, 0)
    else:
        num_bins = calc_best_num_of_hist_bins(container.number_of_bytes, container.number_of_masking_bytes)  # or 0 ('auto')
    hist_boundary = [0, num_bins-1]

    chi2test_engine = Chi2TestEngine(params['engine_name'],
                                     partition_function,
                                     n_bins=num_bins,
                                     bin_range=hist_boundary)

    # We choose here to plot the resulting curve
    session = Session(container,
                      engine=chi2test_engine,
                      output_method=SingleVectorPlotOutputMethod(
                          figure_params_along_time=params['figure_params_along_time'],
                          figure_params_along_trace=params['figure_params_along_trace'],
                          output_path='./results/chi2-test'),
                      output_steps=params['batch_size'])
    session.run(batch_size=params['batch_size'])


if __name__ == '__main__':
    chi2_test(chi2_test_config, fixed_random_traces)
