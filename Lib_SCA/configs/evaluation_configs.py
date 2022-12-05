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
    'figure_params': {'title': 'ttest_result', 'x_label': 'time', 'y_label': 't_score'},
}

"""
continuous mutual information
"""
cmi_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'normal',

    # the total number of key guesses used in the cpa attack
    # the index of the correct key guess
    'no_of_key_guesses': 256,
    'idx_of_correct_key_guess': 0,

    # engine name (trivial)
    'engine_name': 'cmi',

    # attack range
    # attack specific time range of the generated traces (python generator)
    'attack_range': range(0, 4),

    # number of shuffles
    'num_shuffles': 25,

    # batch size
    'batch_size': 1000000,

    # plotting params
    'figure_params': {'title': ['cmi+mi', 'cmi+pv'], 'x_label': ['time', 'time'], 'y_label': ['mi', 'pv']}
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
    'figure_params': {'title': 'graph_test_result', 'x_label': 'time', 'y_label': '***'}
}
