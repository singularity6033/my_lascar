"""
In this script, we show how to perform side-channel characterisation using Welch's T-test to study the behaviour of an Aes Sbox

The characterisation is made with the TTestEngine
Its constructor needs a partition function, which will separate leakages into two classes.

"""
from tqdm import tqdm
from Lib_SCA.config_extractor import JSONConfig
from configs.evaluation_configs import chi2_test_config
from configs.simulation_configs import fixed_random_traces
from Lib_SCA.lascar import SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import SingleVectorPlotOutputMethod
from Lib_SCA.lascar import Session, Chi2TestEngine
from real_traces_generator import real_trace_container


def calc_upper_bound(no_of_bytes, no_of_masking_bytes):
    return no_of_bytes * (no_of_masking_bytes + 1) * 8


def chi2_test(params, trace_params, output_path):
    container = None
    if params['mode'] == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_params=trace_params)
        if not container.masking:
            num_bins = calc_upper_bound(container.number_of_bytes, 0)
        else:
            num_bins = calc_upper_bound(container.number_of_bytes, container.number_of_masking_bytes)
        offsets = [-3 * container.noise_sigma_el, 3 * container.noise_sigma_el]
        hist_boundary = [0, num_bins] + offsets
    elif params['mode'] == 'real':
        container, t_info = real_trace_container(dataset_path=params['dataset_path'],
                                                 num_traces=1000,
                                                 t_start=0,
                                                 t_end=1262)
        hist_boundary = [t_info['min_leakage'], t_info['max_leakage']]

    num_bins = int(hist_boundary[1] - hist_boundary[0]) // params['bin_size']

    def partition_function(value):
        # partition_function must take 1 argument: the value returned by the container at each trace
        # fix and random sets have already been partitioned in container
        return int(value["trace_idx"] % 2 == 0)

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
                          output_path=output_path,
                          filename=params['engine_name']),
                      output_steps=params['batch_size'])
    session.run(batch_size=params['batch_size'])

    del chi2test_engine
    del session


if __name__ == '__main__':
    # chi2_test(chi2_test_config, fixed_random_traces, output_path='./results/chi2-test')
    chi2_test(chi2_test_config, None, output_path='./results/chi2-test')
    gt_params = chi2_test_config
    trace_info = fixed_random_traces
    # json config file generation
    json_config = JSONConfig('chi2test_v6')
    # 500k
    # for m_noise_sigma_el in [0, 0.25, 0.5, 1]:
    #     for m_masking_bytes in range(10):
    #         trace_info['noise_sigma_el'] = m_noise_sigma_el
    #         trace_info['number_of_masking_bytes'] = m_masking_bytes
    #         trace_info['_id'] = '#mask_' + str(m_masking_bytes) + '_el_' + str(m_noise_sigma_el) + \
    #                             '_#trace_' + str(trace_info['number_of_traces'] // 1000) + 'k'
    #         json_config.generate_config(trace_info)

    '''for m_number_of_traces in [50000, 100000, 250000, 350000]:
        for m_noise_sigma_el in [0, 0.25, 0.5, 1, 1.5, 2]:
            for m_masking_bytes in range(10):
                trace_info['number_of_traces'] = m_number_of_traces
                trace_info['noise_sigma_el'] = m_noise_sigma_el
                trace_info['number_of_masking_bytes'] = m_masking_bytes
                trace_info['_id'] = 'el_' + str(m_noise_sigma_el) + \
                                    '_#mask_' + str(m_masking_bytes) + \
                                    '_#trace_' + str(trace_info['number_of_traces'] // 1000) + 'k'
                json_config.generate_config(trace_info)'''

    # get json config file
    '''dict_list = json_config.get_config()
    for i, dict_i in tqdm(enumerate(dict_list)):
        print('[INFO] Processing #', i)
        chi2_test(gt_params, dict_i, output_path='./results/chi2test_v6/' + dict_i['_id'])'''
