import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
from tqdm import tqdm

from Lib_SCA.config_extractor import YAMLConfig, JSONConfig
from Lib_SCA.configs.evaluation_configs import graph_distance_config
from Lib_SCA.configs.simulation_configs import fixed_random_traces, normal_simulated_traces
from Lib_SCA.lascar import SimulatedPowerTraceContainer, SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import SingleOnePlotOutputMethod
from Lib_SCA.lascar import Session, GraphTestEngine, GraphMIEngine, GraphDistanceEngine, GraphTestEngine_Attack
from Lib_SCA.lascar.tools.aes import sbox


def graph_based_distance(params, trace_params):
    if params['mode'] == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_params=trace_params)
    elif params['mode'] == 'real':
        pass

    def partition_function(value):
        # partition_function must take 1 argument: the value returned by the container at each trace
        # fix and random sets have already been partitioned in container
        return int(value["trace_idx"] % 2 != 0)

    graph_distance_engine = GraphDistanceEngine(params['engine_name'],
                                                partition_function,
                                                time_delay=2,
                                                dim=3,
                                                distance_type=params['distance_type'],
                                                test_type=params['test_type'])

    output_path = os.sep.join(['./results/graph_distance', params['_id']])
    # We choose here to plot the resulting curve
    session = Session(container, engine=graph_distance_engine,
                      output_method=SingleOnePlotOutputMethod(figure_params=params['figure_params'],
                                                              output_path=output_path,
                                                              filename=params['_id'],
                                                              display=False),
                      output_steps=params['batch_size'])
    session.run(batch_size=params['batch_size'])

    del graph_distance_engine
    del session


if __name__ == '__main__':
    gt_params = graph_distance_config
    trace_info = fixed_random_traces
    # json config file generation
    json_config = JSONConfig('graph_dist_params_0')
    # 10k, 50k, 200k, 500k, 1000k
    for dist in ['edit_distance', 'vertex_edge_overlap', 'vertex_edge_distance', 'lambda_dist', 'deltacon0']:
        for test_type in ['z-test', 't-test', 'chi-test', 'ks_2samp', 'cramervonmises_2samp']:
            gt_params['distance_type'] = dist
            gt_params['test_type'] = test_type
            gt_params['_id'] = dist + '+' + test_type
            json_config.generate_config(gt_params)

    # get json config file
    dict_list = json_config.get_config()
    for i, dict_i in tqdm(enumerate(dict_list)):
        print('[INFO] Processing #', i)
        graph_based_distance(dict_i, trace_info)

    # graph_based_distance(gt_params, trace_info)
