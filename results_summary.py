import os
import pandas as pd
import numpy as np
from tqdm import tqdm

data_path = './results/graph_direct_test_all_in_one_dist_v1'
filenames = os.listdir(data_path)

rows = ['corr', 'dist']
columns = ['2', '3', '5']

dic = dict()
traces = ['500', '750', '1000', '2000', '3000', '4000', '5000', '7500', '10000',
          '20000', '50000', '100000', '200000', '500000', '1000000']
# for trace in traces:
#     dic[trace] = [[1] * len(traces) for _ in range(len(columns))]

res = [[1] * len(traces) for _ in range(len(columns))]

for filename in tqdm(filenames):
    items = filename.split('_')

    # row = items[1]
    # r_idx = rows.index(row)

    col = items[0].lstrip('#r')
    c_idx = columns.index(col)
    #
    # el = items[0].split('_')[1]
    # el_idx = columns.index(el)

    trace = items[-1]
    t_idx = traces.index(trace)
    # if trace not in traces:
    #     continue
    df = pd.read_excel(os.sep.join([data_path, filename, 'one_result', 'tables', 'graph_direct_test_aio.xlsx']), header=None)
    data = np.array(df)[1:, 1:]
    # graph_data = np.array(df)[1, 1]
    # data = np.max(cmi_data)
    # dic[trace][mask_idx][el_idx] = [float(data[0]), float(data[-1])]
    # dic[trace][mask_idx][el_idx] = float(data)
    res[c_idx][t_idx] = np.min(data)
# new_columns = columns[::-1] + [' ']

with pd.ExcelWriter(os.sep.join([data_path, 'graph_direct_test_all_in_one_dist_v1.xlsx'])) as writer:
    # for trace in traces:
    #     df = pd.DataFrame(np.array(dic[trace], ndmin=2), columns=columns)
    #     df.to_excel(writer, sheet_name=trace)
    df = pd.DataFrame(np.array(res, ndmin=2), columns=traces)
    df.to_excel(writer)
