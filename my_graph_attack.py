import json_config as json_config
import numpy as np
from tqdm import tqdm

from Lib_SCA.config_extractor import JSONConfig
from Lib_SCA.lascar.tools.aes import sbox
from Lib_SCA.lascar import SimulatedPowerTraceStandardContainer, SimulatedPowerTraceContainer
from Lib_SCA.lascar import SingleVectorPlotOutputMethod, SingleMatrixPlotOutputMethod
from Lib_SCA.lascar import Session, GraphAttackEngineTraceAllCorr_SB, GraphAttackEngineTraceAllKSDist_SB, \
    GraphAttackEngineTraceAllChi2_SB
from configs.simulation_configs import normal_simulated_traces
from configs.attack_configs import graph_attack_aio_config
from real_traces_generator import real_trace_container, real_trace_container_random


def graph_attack_aio(params, trace_params, output_path):
    container = None
    if params['mode'] == 'normal':
        container = SimulatedPowerTraceContainer(config_params=trace_params)
    elif params['mode'] == 'real':
        container, t_info = real_trace_container_random(dataset_path=params['dataset_path'],
                                                        num_traces=params['num_traces'],
                                                        t_start=0,
                                                        t_end=1262)

    attack_byte = params['attack_byte']
    guess_range = range(params['no_of_key_guesses'])

    # selection attack regions along time axis
    # container.leakage_section = params['attack_range']

    def selection_function(value, guess, bit, ab=attack_byte):
        return sbox[value["plaintexts"][ab] ^ guess] & bit

    if params['graph_type'] == 'corr':
        graph_attack_aio_engine = GraphAttackEngineTraceAllCorr_SB(params['engine_name'],
                                                                   selection_function,
                                                                   guess_range,
                                                                   solution=-1,
                                                                   k=params['k'],
                                                                   mode=params['d_mode'])
    elif params['graph_type'] == 'dist':
        graph_attack_aio_engine = GraphAttackEngineTraceAllKSDist_SB(params['engine_name'],
                                                                     selection_function,
                                                                     guess_range,
                                                                     num_bins=params['num_bins'],
                                                                     solution=-1,
                                                                     k=params['k'],
                                                                     mode=params['d_mode'])
    elif params['graph_type'] == 'chi2':
        graph_attack_aio_engine = GraphAttackEngineTraceAllChi2_SB(params['engine_name'],
                                                                   selection_function,
                                                                   guess_range,
                                                                   num_bins=params['num_bins'],
                                                                   solution=-1,
                                                                   k=params['k'],
                                                                   mode=params['d_mode'])

    # We choose here to plot the resulting curve
    session = Session(container,
                      engine=graph_attack_aio_engine,
                      output_method=SingleMatrixPlotOutputMethod(
                          figure_params_along_time=params['figure_params_along_time'],
                          figure_params_along_trace=params['figure_params_along_trace'],
                          output_path=output_path,
                          filename=params['engine_name']
                      ))
    session.run(batch_size=params['batch_size'])

    del graph_attack_aio_engine
    del session


if __name__ == '__main__':
    trace_info = normal_simulated_traces

    # graph_attack_aio_config['dataset_path'] = dataset_path
    engine_configs = graph_attack_aio_config
    # graph_attack_aio(graph_attack_aio_config, trace_info, output_path='./results_attack/yzs')

    json_config_engine = JSONConfig('graph_attack_fr_v1_chi2_sb')
    num_traces = [200000]

    for m in num_traces:
        for gt in ['chi2']:
            for d_mode in ['edit', 'sd_l2', 'sd_lmax']:
                engine_configs['num_traces'] = m
                engine_configs['graph_type'] = gt
                engine_configs['d_mode'] = d_mode
                engine_configs['_id'] = '#gt_' + gt + '_#dmode_' + d_mode + '_#trace_' + str(m // 2000) + 'k'
                json_config_engine.generate_config(engine_configs)

    configs = json_config_engine.get_config()
    for i, config in tqdm(enumerate(configs)):
        print('[INFO] Processing #', i)
        graph_attack_aio(config, trace_info,
                         output_path='./results_attack/graph_attack/fr_v1_chi2_sb/' + config['_id'])
