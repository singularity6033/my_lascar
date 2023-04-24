import numpy as np
from Lib_SCA.hdf5_files_import import read_hdf5_proj
from Lib_SCA.lascar import TraceBatchContainer
from tqdm import tqdm


def real_trace_container(dataset_path, num_traces, t_start, t_end, offset=0):
    traces, plaintexts, ciphertexts = read_hdf5_proj(database_file=dataset_path,
                                                     idx_srt=0 + offset,
                                                     idx_end=num_traces + offset,
                                                     start=t_start,
                                                     end=t_end,
                                                     load_trace=True,
                                                     load_plaintext=True,
                                                     load_ciphertext=True)

    # construct leakages and values
    leakages = traces
    trace_info = {'max_leakage': np.max(leakages),
                  'min_leakage': np.min(leakages)}

    value_dtype = np.dtype([('plaintexts', np.uint8, plaintexts[0, :].shape),
                            ('ciphertexts', np.uint8, ciphertexts[0, :].shape),
                            ('trace_idx', np.uint8, ())])
    value = np.zeros((), dtype=value_dtype)

    values_dtype = np.dtype((value_dtype, (num_traces, )))
    values = np.zeros((), dtype=values_dtype)

    print("[INFO] dataset loading...")
    for i in tqdm(range(num_traces)):
        value['trace_idx'] = i
        value['plaintexts'] = plaintexts[i, :]
        value['ciphertexts'] = ciphertexts[i, :]
        values[i] = value
        # values = value if i == 0 else np.hstack([tmp, value])  # this way is too time consuming

    container = TraceBatchContainer(leakages, values), trace_info

    return container


def real_trace_container_random(dataset_path, num_traces, t_start, t_end, offset=0):
    traces, plaintexts, ciphertexts = read_hdf5_proj(database_file=dataset_path,
                                                     idx_srt=0 + offset,
                                                     idx_end=num_traces + offset,
                                                     start=t_start,
                                                     end=t_end,
                                                     load_trace=True,
                                                     load_plaintext=True,
                                                     load_ciphertext=True)

    # construct leakages and values
    leakages_random = traces[1::2]
    trace_info = {'max_leakage': np.max(leakages_random),
                  'min_leakage': np.min(leakages_random)}

    value_dtype = np.dtype([('plaintexts', np.uint8, plaintexts[0, :].shape),
                            ('ciphertexts', np.uint8, ciphertexts[0, :].shape)])
    value = np.zeros((), dtype=value_dtype)

    values_dtype = np.dtype((value_dtype, (leakages_random.shape[0], )))
    values = np.zeros((), dtype=values_dtype)

    idx = 0
    print("[INFO] dataset loading...")
    for i in tqdm(range(1, num_traces, 2)):
        value['plaintexts'] = plaintexts[i, :]
        value['ciphertexts'] = ciphertexts[i, :]
        values[idx] = value
        idx += 1
        # values = value if i == 0 else np.hstack([tmp, value])  # this way is too time consuming

    container = TraceBatchContainer(leakages_random, values), trace_info

    return container


if __name__ == '__main__':
    x = real_trace_container_random('./sca_real_data/EM_Sync_TVLA_1M.sx', 1000, 0, 1262)
    v = x[0][:].values

