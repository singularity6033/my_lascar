"""
t-test
"""
t_test_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'real',  # or real

    'dataset_path': '/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx',
    'num_traces': 1000,
    # used in 'real' mode

    # engine name (trivial)
    'engine_name': 'ttest',

    # batch size
    'batch_size': 50000,

    # plotting params
    'figure_params_along_time': {'title': 't-test_result', 'x_label': 'time', 'y_label': 'p-value'},
    'figure_params_along_trace': {'title': 't-test_result', 'x_label': 'trace_no', 'y_label': 'p-value'}
}

"""
chi2-test
"""
chi2_test_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'real',
    'num_bins': 10,

    'dataset_path': '/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx',
    'num_traces': 1000000,
    # used in 'real' mode

    # engine name (trivial)
    'engine_name': 'chi2test',

    # batch size
    'batch_size': 50000,

    # plotting params
    'figure_params_along_time': {'title': 'chi2-test_result', 'x_label': 'time', 'y_label': 'p-value'},
    'figure_params_along_trace': {'title': 'chi2-test_result', 'x_label': 'trace_no', 'y_label': 'p-value'}
}

"""
continuous mutual information
"""
cmi_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'real',
    'num_bins': 8,

    'dataset_path': '/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx',
    'num_traces': 1000,
    'attack_byte': 0,

    # the total number of key guesses used in the cpa attack
    # the index of the correct key guess
    'no_of_key_guesses': 34,
    'idx_of_correct_key_guess': 0,

    # engine name (trivial)
    'engine_name': 'cmi',

    # attack range
    # attack specific time range of the generated traces (python generator)
    # 'attack_range': range(0, 4),

    # number of shuffles
    'num_shuffles': 10,

    # batch size
    'batch_size': 50000,

    # plotting params
    'figure_params_along_time': {'title': ['cmi+mi', 'cmi+pv'], 'x_label': ['time', 'time'], 'y_label': ['mi', 'pv']},
    'figure_params_along_trace': {'title': ['cmi+mi', 'cmi+pv'], 'x_label': ['trace_batch', 'trace_batch'],
                                  'y_label': ['mi', 'pv']}
}

""" 
graph-based test
"""
graph_direct_test_aio_config = {
    # the type of the container (fixed-random trace or real trace)
    'mode': 'real',
    'dataset_path': '/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx',
    'num_traces': 1000,

    'type': 'dist',

    'k': 3,

    # engine name (trivial)
    'engine_name': 'graph_direct_test_aio',

    # batch size
    'batch_size': 1000000,

    # plotting params
    'figure_params': {'title': 'graph_direct_test_aio', 'x_label': 'trace_batch', 'y_label': 'p-value'}
}

graph_direct_test_along_time_psr_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'real',
    'dataset_path': '/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx',
    'num_traces': 1000,

    'time_delay': 1,
    'dim': 10,
    'sampling_interval': 1,

    'type': 'corr',

    # engine name (trivial)
    'engine_name': 'graph_direct_test_along_time_psr',

    # batch size
    'batch_size': 50000,

    # plotting params
    'figure_params': {'title': 'graph_direct_test_along_time_psr', 'x_label': 'trace_batch', 'y_label': 'p-value'}
}

graph_direct_test_along_time_amp_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'real',
    'dataset_path': '/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx',
    'num_traces': 1000,

    'num_of_amp_groups': 10,
    'num_of_moments': 4,

    'type': 'corr',

    # engine name (trivial)
    'engine_name': 'graph_direct_test_along_time_amp',

    # batch size
    'batch_size': 50000,

    # plotting params
    'figure_params': {'title': 'graph_direct_test_along_time_amp', 'x_label': 'trace_batch', 'y_label': 'p-value'}
}

graph_direct_test_along_trace_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'real',
    'dataset_path': '/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx',
    'num_traces': 1000,

    # engine name (trivial)
    'engine_name': 'graph_direct_test_along_trace',

    # batch size
    'batch_size': 50,

    # plotting params
    'figure_params': {'title': 'graph_direct_test_along_trace', 'x_label': 'trace_batch', 'y_label': 'p-value'}
}

graph_direct_test_attack_psr_config = {
    # the type of the container (normal trace or real trace)
    'mode': 'real',
    'dataset_path': '/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx',
    'num_traces': 1000000,

    'time_delay': 2,
    'dim': 10,
    'sampling_interval': 1,

    'measurement': 'corr',

    'no_of_key_guesses': 256,
    'idx_of_correct_key_guess': 0,

    'attack_byte': 0,

    # engine name (trivial)
    'engine_name': 'graph_direct_test_attack_psr',

    # batch size
    'batch_size': 50000,

    # plotting params
    'figure_params': {'title': 'graph_direct_test_attack_psr', 'x_label': 'key_guess', 'y_label': 'p-value'}
}

graph_direct_test_attack_amp_config = {
    # the type of the container (normal trace or real trace)
    'mode': 'real',
    'dataset_path': '/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx',
    'num_traces': 1000,

    'num_of_amp_groups': 10,
    'num_of_moments': 4,

    'measurement': 'corr',

    'no_of_key_guesses': 256,
    'idx_of_correct_key_guess': 0,

    'attack_byte': 0,

    # engine name (trivial)
    'engine_name': 'graph_direct_test_attack_amp',

    # batch size
    'batch_size': 50000,

    # plotting params
    'figure_params': {'title': 'graph_direct_test_attack_amp', 'x_label': 'key_guess', 'y_label': 'p-value'}
}

graph_distance_test_along_time_psr_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'real',

    'dataset_path': '/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx',
    'num_traces': 1000,

    'time_delay': 2,
    'dim': 10,
    'sampling_interval': 1,
    'measurement': 'corr',

    'distance_type': 'lambda_dist',
    'num_bins': 50,

    # engine name (trivial)
    'engine_name': 'graph_distance_test_along_time_psr',

    # batch size
    'batch_size': 50000,

    # plotting params
    'figure_params': {'title': 'graph_distance_test_along_time_psr', 'x_label': 'trace_batch', 'y_label': 'p-value'}
}

graph_distance_test_along_time_amp_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'real',

    'dataset_path': '/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx',
    'num_traces': 1000,

    'num_of_amp_groups': 10,
    'num_of_moments': 4,
    'measurement': 'corr',

    'distance_type': 'lambda_dist',
    'num_bins': 50,

    # engine name (trivial)
    'engine_name': 'graph_distance_test_along_time_amp',

    # batch size
    'batch_size': 50000,

    # plotting params
    'figure_params': {'title': 'graph_distance_test_along_time_amp', 'x_label': 'trace_batch', 'y_label': 'p-value'}
}

graph_distance_test_along_trace_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'real',

    'dataset_path': '/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx',
    'num_traces': 1000,

    'distance_type': 'lambda_dist',
    'num_bins': 50,

    # engine name (trivial)
    'engine_name': 'graph_distance_trace_based',

    # batch size
    'batch_size': 1000,

    # plotting params
    'figure_params': {'title': 'graph_distance_trace_based_result', 'x_label': 'trace_batch', 'y_label': 'p-value'}
}

graph_distance_test_attack_psr_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'real',

    'dataset_path': '/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx',
    'num_traces': 1000,

    'time_delay': 2,
    'dim': 10,
    'sampling_interval': 1,
    'measurement': 'corr',

    'distance_type': 'lambda_dist',
    'num_bins': 50,

    'no_of_key_guesses': 256,
    'idx_of_correct_key_guess': 0,

    'attack_byte': 0,

    # engine name (trivial)
    'engine_name': 'graph_distance_test_attack_psr',

    # batch size
    'batch_size': 50000,

    # plotting params
    'figure_params': {'title': 'graph_distance_test_attack_psr', 'x_label': 'key_guess', 'y_label': 'p-value'}
}

graph_distance_test_attack_amp_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'real',

    'dataset_path': '/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx',
    'num_traces': 1000,

    'num_of_amp_groups': 10,
    'num_of_moments': 4,
    'measurement': 'corr',

    'distance_type': 'lambda_dist',
    'num_bins': 50,

    'no_of_key_guesses': 256,
    'idx_of_correct_key_guess': 0,

    'attack_byte': 0,

    # engine name (trivial)
    'engine_name': 'graph_distance_test_attack_amp',

    # batch size
    'batch_size': 50000,

    # plotting params
    'figure_params': {'title': 'graph_distance_test_attack_amp', 'x_label': 'key_guess', 'y_label': 'p-value'}
}
