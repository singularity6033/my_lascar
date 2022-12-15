"""
cpa
"""
cpa_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'normal',

    # the total number of key guesses used in the cpa attack
    # the index of the correct key guess
    'no_of_key_guesses': 256,
    'idx_of_correct_key_guess': 0,

    # engine name (trivial)
    'engine_name': 'cpa',

    # attack range
    # attack specific time range of the generated traces (python generator)
    'attack_range': range(0, 4),

    # batch size
    'batch_size': 500,

    # plotting params
    'figure_params_along_time': {'title': 'cpa_result', 'x_label': 'time', 'y_label': 'correlation coefficient'},
    'figure_params_along_trace': {'title': 'cpa_result', 'x_label': 'trace_batch', 'y_label': 'correlation coefficient'}

}

"""
dpa
"""
dpa_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'normal',

    # the total number of key guesses used in the cpa attack
    # the index of the correct key guess
    'no_of_key_guesses': 256,
    'idx_of_correct_key_guess': 0,

    # engine name (trivial)
    'engine_name': 'dpa',

    # attack range
    # attack specific time range of the generated traces (python generator)
    'attack_range': range(0, 4),

    # batch size
    'batch_size': 1000,

    # plotting params
    'figure_params': {'title': 'dpa_result', 'x_label': 'time', 'y_label': 'dom'}
}
