import h5py
import numpy as np
import os.path
import sys
import matplotlib.pyplot as plt

from .sca_DataKeys import ProjectDataSetTags


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


def read_hdf5_proj(database_file,
                   idx_srt=None,
                   idx_end=None,
                   start=None,
                   end=None,
                   load_trace=False,
                   load_plaintext=False,
                   load_ciphertext=False):
    check_file_exists(database_file)

    try:
        in_file = h5py.File(database_file, "r")
        key = in_file[ProjectDataSetTags.KEY.value][idx_srt:idx_end]
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % database_file)
        sys.exit(-1)
    traces = []
    if load_trace:
        traces = np.array(in_file[ProjectDataSetTags.TRACES.value][idx_srt:idx_end, start:end]).astype('float64')

    if load_plaintext and load_ciphertext:
        return traces, in_file[ProjectDataSetTags.PLAIN_TEXT.value][idx_srt:idx_end], in_file[ProjectDataSetTags.CIPHER_TEXT.value][
            idx_srt:idx_end]
    elif load_plaintext and not load_ciphertext:
        return traces, in_file[ProjectDataSetTags.PLAIN_TEXT.value][idx_srt:idx_end]
    elif not load_plaintext and load_ciphertext:
        return traces, in_file[ProjectDataSetTags.CIPHER_TEXT.value][idx_srt:idx_end]
    else:
        return traces
