import os
import json

import numpy as np
from tqdm import tqdm

from Lib_SCA.lascar import SimulatedPowerTraceContainer
from configs.simulation_configs import normal_simulated_traces

base_save_path = './results/data_to_xyc/'
filename = 'test1'

with open(os.sep.join([base_save_path, 'configs', filename + '.json']), 'a') as f:
    json.dump(normal_simulated_traces, f)
    f.write("\n")
f.close()

container = SimulatedPowerTraceContainer(config_params=normal_simulated_traces)


attack_time = container.attack_sample_point
exp_byte = container.idx_exp[0]

tmp = {}
data_path = os.sep.join([base_save_path, 'json', filename + '.json'])
for trace_i in tqdm(container):
    tmp['leakages'] = list(trace_i.leakage)
    tmp['plaintext'] = int(trace_i.value['plaintext'][exp_byte][attack_time])
    with open(data_path, 'a') as f:
        json.dump(tmp, f)
        f.write("\n")
f.close()
