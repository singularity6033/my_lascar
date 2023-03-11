"""
t-test
"""
t_test_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'real',  # or real

    'dataset_path': '/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx',
    # used in 'real' mode

    # engine name (trivial)
    'engine_name': 'ttest',

    # batch size
    'batch_size': 500,

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
    'bin_size': 1,

    'dataset_path': '/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx',
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
    'mode': 'normal',
    'bin_size': 1,

    # the total number of key guesses used in the cpa attack
    # the index of the correct key guess
    'no_of_key_guesses': 5,
    'idx_of_correct_key_guess': 0,

    # engine name (trivial)
    'engine_name': 'cmi',

    # attack range
    # attack specific time range of the generated traces (python generator)
    'attack_range': range(0, 4),

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
graph_test_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'fix_random',

    # engine name (trivial)
    'engine_name': 'graph_test',

    # batch size
    'batch_size': 50000,

    # plotting params
    'figure_params': {'title': 'graph_test_result', 'x_label': 'trace_batch', 'y_label': 'p-value'}
}

graph_test_trace_based_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'fix_random',

    # engine name (trivial)
    'engine_name': 'graph_test_trace_based',

    # batch size
    'batch_size': 50,

    # plotting params
    'figure_params': {'title': 'graph_test_result', 'x_label': 'trace_batch', 'y_label': 'p-value'}
}

graph_test_attack_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'normal',

    'no_of_key_guesses': 5,
    'idx_of_correct_key_guess': 0,

    # engine name (trivial)
    'engine_name': 'graph_test_attack',

    # batch size
    'batch_size': 50000,

    # plotting params
    'figure_params': {'title': 'graph_test_attack', 'x_label': 'key_guess', 'y_label': 'p-value'}
}

graph_distance_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'fix_random',

    'distance_type': 'lambda_dist',
    'num_bins': 50,

    # engine name (trivial)
    'engine_name': 'graph_distance',

    # batch size
    'batch_size': 50000,

    # plotting params
    'figure_params': {'title': 'graph_distance_result', 'x_label': 'trace_batch', 'y_label': 'p-value'}
}

graph_distance_trace_based_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'fix_random',

    'distance_type': 'lambda_dist',
    'num_bins': 50,

    # engine name (trivial)
    'engine_name': 'graph_distance_trace_based',

    # batch size
    'batch_size': 50,

    # plotting params
    'figure_params': {'title': 'graph_distance_trace_based_result', 'x_label': 'trace_batch', 'y_label': 'p-value'}
}

graph_distance_attack_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'normal',

    'no_of_key_guesses': 5,
    'idx_of_correct_key_guess': 0,

    'distance_type': 'deltacon0',

    # engine name (trivial)
    'engine_name': 'graph_distance_attack',

    # batch size
    'batch_size': 5000,

    # plotting params
    'figure_params': {'title': 'graph_distance_attack_result', 'x_label': 'key_guess', 'y_label': 'p-value'}
}
