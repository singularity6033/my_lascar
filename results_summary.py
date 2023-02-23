import os
import pandas as pd
import numpy as np
from tqdm import tqdm

data_path = './results/chi2test_v4'
filenames = os.listdir(data_path)

rows = ['0', '0.25', '0.5', '1']
columns = [str(i) for i in range(10)]
dic = dict()

for trace in ['50k', '100k', '250k', '500k']:
    dic[trace] = np.zeros((len(rows), len(columns)))

for filename in tqdm(filenames):
    items = filename.split('_#')
    el = items[0].split('_')[1]
    el_idx = rows.index(el)
    mask = items[1].split('_')[1]
    mask_idx = columns.index(mask)
    trace = items[2].split('_')[1]
    df = pd.read_excel(os.sep.join([data_path, filename, 'along_time', 'tables', 'chi2test.xlsx']), header=None)
    data = np.min(np.array(df)[1:, 1:])
    dic[trace][el_idx][mask_idx] = data

new_columns = columns[::-1] + [' ']

with pd.ExcelWriter(os.sep.join([data_path, 'summary.xlsx'])) as writer:
    for trace in ['50k', '100k', '250k', '500k']:
        df = pd.DataFrame(np.concatenate((np.array(rows, ndmin=2).T, dic[trace]), axis=1), columns=new_columns[::-1])
        df.to_excel(writer, sheet_name=trace)



