"""
t-test
"""
t_test_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'fix_random',

    # engine name (trivial)
    'engine_name': 'ttest',

    # batch size
    'batch_size': 1000,

    # plotting params
    'figure_params_along_time': {'title': 't-test_result', 'x_label': 'time', 'y_label': 't-score'},
    'figure_params_along_trace': {'title': 't-test_result', 'x_label': 'trace_batch', 'y_label': 't-score'}
}

"""
continuous mutual information
"""
cmi_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'normal',

    # the total number of key guesses used in the cpa attack
    # the index of the correct key guess
    'no_of_key_guesses': 3,
    'idx_of_correct_key_guess': 0,

    # engine name (trivial)
    'engine_name': 'cmi',

    # attack range
    # attack specific time range of the generated traces (python generator)
    'attack_range': range(0, 4),

    # number of shuffles
    'num_shuffles': 2,

    # batch size
    'batch_size': 50000,

    # plotting params
    'figure_params_along_time': {'title': ['cmi+mi', 'cmi+pv'], 'x_label': ['time', 'time'], 'y_label': ['mi', 'pv']},
    'figure_params_along_trace': {'title': ['cmi+mi', 'cmi+pv'], 'x_label': ['trace_batch', 'trace_batch'], 'y_label': ['mi', 'pv']}
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
    'batch_size': 1000,

    # plotting params
    'figure_params': {'title': 'graph_test_result', 'x_label': 'trace_batch', 'y_label': 'p-value'}
}

graph_test_attack_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'normal',

    # engine name (trivial)
    'engine_name': 'graph_test_attack',

    # batch size
    'batch_size': 1000000,

    # plotting params
    'figure_params': {'title': 'graph_test_attack', 'x_label': 'key_guess', 'y_label': 'p-value'}
}

graph_distance_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'fix_random',

    'distance_type': 'deltacon0',
    'test_type': 'z-test',

    # engine name (trivial)
    'engine_name': 'graph_distance',

    # batch size
    'batch_size': 1000,

    # plotting params
    'figure_params': {'title': 'graph_distance_result', 'x_label': 'trace_batch', 'y_label': 'p-value'}
}
