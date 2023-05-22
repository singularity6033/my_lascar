import numpy as np

"""
cpa
"""
cpa_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'real',

    'dataset_path': './sca_real_data/EM_Sync_TVLA_1M.sx',
    'num_traces': 1000000,

    'attack_byte': 2,

    # the total number of key guesses used in the cpa attack
    # the index of the correct key guess
    'no_of_key_guesses': 256,
    'idx_of_correct_key_guess': 2,

    # engine name (trivial)
    'engine_name': 'cpa',

    # attack range
    # attack specific time range of the generated traces (python generator)
    'attack_range': range(0, 4),

    # batch size
    'batch_size': 500000,

    # plotting params
    'figure_params_along_time': {'title': 'cpa_result', 'x_label': 'time', 'y_label': 'correlation coefficient'},
    'figure_params_along_trace': {'title': 'cpa_result', 'x_label': 'trace_no', 'y_label': 'correlation coefficient'}

}

"""
dpa
"""
dpa_config = {
    # the type of the container (normal trace or fixed-random trace or real trace)
    'mode': 'normal',
    'dataset_path': '/media/mldadmin/home/s122mdg34_05/my_lascar/sca_real_data/EM_Sync_TVLA_1M.sx',
    'num_traces': 200000,

    'attack_byte': 0,

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
    'batch_size': 500000,

    # plotting params
    # plotting params
    'figure_params_along_time': {'title': 'dpa_result', 'x_label': 'time', 'y_label': 'dom'},
    'figure_params_along_trace': {'title': 'dpa_result', 'x_label': 'trace_no', 'y_label': 'dom'}
}

"""
graph based attack
"""
graph_attack_aio_config = {
    'mode': 'real',
    # 'dataset_path': './sca_real_data/EM_Sync_TVLA_1M.sx',
    # 'dataset_path': './sca_real_data/EM_TVLA_1M.sx',
    'dataset_path': './sca_real_data/dataset_from_sca_toolkit/ascad/ascad.sx',

    'num_traces': 200000,
    'attack_byte': 2,
    'no_of_key_guesses': 256,

    'graph_type': 'corr',
    ## spectral_dist, vertex_edge_dist, corr_dist, chi2_dist, deltacon0_dist, resistance_dist
    'dist_type': 'chi2_dist',

    # if use spectral_dist
    'sd_params': {'k': 10,
                  'p': 2,  # p-norm ex: 1, np.inf'
                  'kind': 'adjacency',  # 'laplacian', 'laplacian_norm', 'adjacency'
                  },

    # use in histogram estimation
    'num_bins': 100,

    # engine name (trivial)
    'engine_name': 'graph_attack_aio',

    # batch size
    'batch_size': 10000,

    # plotting params
    'figure_params_along_time': {'title': 'graph_attack_aio', 'x_label': 'time', 'y_label': 'distance'},
    'figure_params_along_trace': {'title': 'graph_attack_aio', 'x_label': 'trace_no', 'y_label': 'distance'}
}
