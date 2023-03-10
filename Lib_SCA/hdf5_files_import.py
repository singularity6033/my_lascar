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
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % database_file)
        sys.exit(-1)
    traces = []
    idx = np.arange(idx_srt, idx_end, dtype=np.uint32)
    idx = idx.tolist()
    if load_trace:
        traces = np.array(in_file[ProjectDataSetTags.TRACES.value][idx, start:end]).astype('float64')

    if load_plaintext and load_ciphertext:
        return traces, in_file[ProjectDataSetTags.PLAIN_TEXT.value][idx], in_file[ProjectDataSetTags.CIPHER_TEXT.value][
            idx]
    elif load_plaintext and not load_ciphertext:
        return traces, in_file[ProjectDataSetTags.PLAIN_TEXT.value][idx]
    elif not load_plaintext and load_ciphertext:
        return traces, in_file[ProjectDataSetTags.CIPHER_TEXT.value][idx]
    else:
        return traces
