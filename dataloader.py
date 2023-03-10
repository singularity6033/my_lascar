import os
import json

import numpy as np
from tqdm import tqdm

from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer, SimulatedPowerTraceContainerWithPlaintext
from configs.simulation_configs import normal_simulated_traces, fixed_random_traces

base_save_path = './results/data_to_xyc/'
filename = 'test5'

config_path = os.sep.join([base_save_path, 'configs', filename + '.json'])
if os.path.isfile(config_path):
    os.remove(config_path)

with open(config_path, 'a') as f:
    json.dump(fixed_random_traces, f)
    f.write("\n")
f.close()

p_text = np.random.randint(0, 256, (20000, 1), np.uint8)
# container = SimulatedPowerTraceContainer(config_params=normal_simulated_traces)
# container = SimulatedPowerTraceFixedRandomContainer(config_params=fixed_random_traces)
container_1 = SimulatedPowerTraceContainerWithPlaintext(config_params=normal_simulated_traces, p_text=p_text, seed=0)
container_2 = SimulatedPowerTraceContainerWithPlaintext(config_params=normal_simulated_traces, p_text=p_text, seed=1)
attack_time = container_1.attack_sample_point
exp_byte = container_1.idx_exp[0]

tmp = {}
data_path = os.sep.join([base_save_path, 'json', filename + '.json'])
if os.path.isfile(data_path):
    os.remove(data_path)

for trace_i_1, trace_i_2 in tqdm(zip(container_1, container_2)):
    tmp['leakages_1'] = list(trace_i_1.leakage)
    tmp['leakages_2'] = list(trace_i_2.leakage)
    if trace_i_1.value['plaintext'][exp_byte][attack_time] == trace_i_2.value['plaintext'][exp_byte][attack_time]:
        tmp['plaintext'] = int(trace_i_1.value['plaintext'][exp_byte][attack_time])
    else:
        print('error')
    with open(data_path, 'a') as f:
        json.dump(tmp, f)
        f.write("\n")
f.close()
