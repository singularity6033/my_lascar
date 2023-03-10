import os
import pandas as pd
import numpy as np
from tqdm import tqdm

data_path = './results/graph_distance_v6'
filenames = os.listdir(data_path)

rows = [str(i) for i in range(10)]
columns = ['0', '0.25', '0.5', '1', '1.5', '2']

dic = dict()

for trace in ['50k', '100k', '250k', '350k']:
    dic[trace] = [[0] * len(columns) for _ in range(len(rows))]

for filename in tqdm(filenames):
    items = filename.split('_#')

    mask = items[1].split('_')[1]
    mask_idx = rows.index(mask)

    el = items[0].split('_')[1]
    el_idx = columns.index(el)

    trace = items[2].split('_')[1]
    df = pd.read_excel(os.sep.join([data_path, filename, 'one_result', 'tables', 'graph_distance.xlsx']), header=None)
    data = np.array(df)[1:, 1:]
    # graph_data = np.array(df)[1, 1]
    # data = np.max(cmi_data)
    # dic[trace][mask_idx][el_idx] = [float(data[0]), float(data[-1])]
    dic[trace][mask_idx][el_idx] = float(data)

# new_columns = columns[::-1] + [' ']

with pd.ExcelWriter(os.sep.join([data_path, 'graph_distance.xlsx'])) as writer:
    for trace in ['50k', '100k', '250k', '350k']:
        df = pd.DataFrame(dic[trace], columns=columns)
        df.to_excel(writer, sheet_name=trace)
