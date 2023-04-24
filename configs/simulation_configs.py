"""
configs for normal_simulated_traces
"""
normal_simulated_traces = {
    # total number of traces
    'number_of_traces': 100000,
    # total number of bytes for each time sample in one trace
    'number_of_bytes': 16,

    # total number of time sample points one trace has
    'number_of_time_samples': 10,

    # position (index) of time sample point we presumptively attack
    'attack_sample_point': 0,

    # linear coefficients of power of exploitable signal and switching noise
    'linear_coefficient_exp': 1,
    'linear_coefficient_switch': 1,

    # index of bytes represented for exploitable signal and switching noise
    # int or list (0-number_of_bytes)
    # all indexes should be exclusive and total number of them should equal to "number_of_bytes"
    'idx_exploitable_bytes': [0],
    'idx_switching_noise_bytes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],

    # parameters of electronic noise (a Multi-Gaussian distribution)
    # mean is a 1-d list with the shape: number_of_time_samples * 1
    # covariance matrix is a 2-d matrix with the shape: number_of_time_samples * number_of_time_samples
    # example (total number of time sample points is 3):
    # noise_sigma_el: [0, 0]
    # noise_mean_el: [[1, 0, 0],
    #                  0, 1, 0],
    #                  0, 0, 1]]
    # note: this is a simple covariance matrix only with variances along diagonal
    # if parameters for all time samples are same, just pass single value, and mean vector and cov matrix will be generated
    # automatically i1 the code
    'noise_mean_el': 0,
    'noise_sigma_el': 0,

    # constant term used in calculation of trace power
    'constant': 0,

    # this parameter used to adjust magnitude of 3 time samples (attack time sample and other 2 after it), where the
    # plaintext (index 0 and 1 are exactly the same but index 2 will have same plaintext for exploitable signal but
    # different for switching noise) has been really encrypted
    'sp_curve': [1, 0.5, 1],

    # this parameter used to adjust linear coefficients of power of both exploitable signal and switching noise
    # if set to true, it will automatically generate a Gaussian-like curve
    'bytes_curve': False,

    # leakage model used in simulation
    # "default" is the hamming weight model
    'leakage_model_name': "default",

    # masking countermeasure
    'masking': False,
    'number_of_masking_bytes': 1,

    # shuffle
    'shuffle': False,
    'shuffle_range': [0, 5],

    # shift
    'shift': False,
    'shift_range': 5
}

""" 
configs for fixed_random_traces  
"""
fixed_random_traces = {
    # total number of traces
    'number_of_traces': 10000,
    # total number of bytes for each time sample in one trace
    'number_of_bytes': 1,

    # number_of_bytes * number_of_time_samples
    # give a list of plaintext for corresponding bytes
    # assume each byte has the same ciphertext along the time axis
    'fixed_set': [6],

    # total number of time sample points one trace has
    'number_of_time_samples': 50,

    # position (index) of time sample point we presumptively attack
    'attack_sample_point': 0,

    # linear coefficients of power of exploitable signal
    'linear_coefficient_exp': 1,

    # index of bytes represented for exploitable signal
    # int or list (0-number_of_bytes)
    # all indexes should be exclusive and total number of them should equal to "number_of_bytes"
    'idx_exploitable_bytes': [0],

    # parameters of electronic noise (a Multi-Gaussian distribution)
    # mean is a 1-d list with the shape: number_of_time_samples * 1
    # covariance matrix is a 2-d matrix with the shape: number_of_time_samples * number_of_time_samples
    # example (total number of time sample points is 3):
    # noise_sigma_el: [0, 0]
    # noise_mean_el: [[1, 0, 0],
    #                  0, 1, 0],
    #                  0, 0, 1]]
    # note: this is a simple covariance matrix only with variances along diagonal
    # if parameters for all time samples are same, just pass single value, and mean vector and cov matrix will be generated
    # automatically i1 the code
    'noise_mean_el': 0,
    'noise_sigma_el': 0.1,

    # constant term used in calculation of trace power
    'constant': 0,

    # this parameter used to adjust magnitude of 3 time samples (attack time sample and other 2 after it), where the
    # plaintext (index 0 and 1 are exactly the same but index 2 will have same plaintext for exploitable signal but
    # different for switching noise) has been really encrypted
    'sp_curve': [1, 0.5, 1],

    # this parameter used to adjust linear coefficients of power of both exploitable signal and switching noise
    # if set to true, it will automatically generate a Gaussian-like curve
    'bytes_curve': False,

    # leakage model used in simulation
    # "default" is the hamming weight model
    'leakage_model_name': "default",

    # masking countermeasure
    'masking': True,
    'number_of_masking_bytes': 0,

    # shuffle
    'shuffle': False,
    'shuffle_range': [0, 5],

    # shift
    'shift': False,
    'shift_range': 5
}
