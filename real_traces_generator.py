import numpy as np
from Lib_SCA.hdf5_files_import import read_hdf5_proj
from Lib_SCA.lascar import TraceBatchContainer
from tqdm import tqdm


def real_trace_container(dataset_path, num_traces, t_start, t_end, offset=0):
    traces, plaintexts, ciphertexts = read_hdf5_proj(database_file=dataset_path,
                                                     idx_srt=0+offset,
                                                     idx_end=num_traces+offset,
                                                     start=t_start,
                                                     end=t_end,
                                                     load_trace=True,
                                                     load_plaintext=True,
                                                     load_ciphertext=True)

    # construct leakages and values
    leakages = traces

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

    container = TraceBatchContainer(leakages, values)

    return container
