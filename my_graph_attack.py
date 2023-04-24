import json_config as json_config
import numpy as np
from tqdm import tqdm

from Lib_SCA.config_extractor import JSONConfig
from Lib_SCA.lascar.tools.aes import sbox
from Lib_SCA.lascar import SimulatedPowerTraceStandardContainer, SimulatedPowerTraceContainer
from Lib_SCA.lascar import SingleVectorPlotOutputMethod, SingleMatrixPlotOutputMethod
from Lib_SCA.lascar import Session, GraphAttackEngineTraceAllCorr, GraphAttackEngineTraceAllDist, \
    GraphAttackEngineTraceBatch
from configs.simulation_configs import normal_simulated_traces
from configs.attack_configs import graph_attack_aio_config
from real_traces_generator import real_trace_container, real_trace_container_random


def graph_attack_aio(params, trace_params, output_path):
    container = None
    if params['mode'] == 'normal':
        container = SimulatedPowerTraceContainer(config_params=trace_params)
    elif params['mode'] == 'real':
        container, t_info = real_trace_container(dataset_path=params['dataset_path'],
                                                 num_traces=params['num_traces'],
                                                 t_start=45400,
                                                 t_end=46100)

    attack_byte = params['attack_byte']
    guess_range = range(params['no_of_key_guesses'])

    # selection attack regions along time axis
    # container.leakage_section = params['attack_range']

    def selection_function(value, guess, bit, ab=attack_byte):
        return sbox[value["plaintexts"][ab] ^ guess] & bit

    # def selection_function(values, guess, ab=attack_byte):
    #     pl = values['plaintext'][:, ab]
    #     sbox_out = sbox[pl ^ guess]
    #     sbox_out_bits = np.unpackbits(sbox_out, axis=1)
    #     hw = (sbox_out_bits == 1).sum(axis=1)
    #     return hw

    if params['graph_type'] == 'corr':
        graph_attack_aio_engine = GraphAttackEngineTraceAllCorr(params['engine_name'],
                                                                selection_function,
                                                                guess_range,
                                                                solution=224,
                                                                k=params['k'],
                                                                mode=params['d_mode'])
    # graph_attack_aio_engine = GraphAttackEngineTraceBatch(params['engine_name'],
    #                                                       selection_function,
    #                                                       guess_range,
    #                                                       solution=0,
    #                                                       graph_type=params['graph_type'],
    #                                                       k=params['k'],
    #                                                       mode=params['d_mode'],
    #                                                       num_bins=params['num_bins'])
    elif params['graph_type'] == 'dist':
        graph_attack_aio_engine = GraphAttackEngineTraceAllDist(params['engine_name'],
                                                                selection_function,
                                                                guess_range,
                                                                num_bins=params['num_bins'],
                                                                solution=0,
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
    graph_attack_aio(graph_attack_aio_config, trace_info, output_path='./results_attack/yzs')

    # json_config_engine = JSONConfig('graph_attack_ascad_v1')
    # num_traces = [1000, 5000, 10000, 60000]
    #
    # for m in num_traces:
    #     for gt in ['corr', 'dist']:
    #         for d_mode in ['l2', 'lmax', 'edit']:
    #             engine_configs['num_traces'] = m
    #             engine_configs['d_mode'] = d_mode
    #             engine_configs['_id'] = '#gt_' + gt + '_#dmode_' + d_mode + '_#trace_' + str(m // 1000) + 'k'
    #             json_config_engine.generate_config(engine_configs)
    #
    # configs = json_config_engine.get_config()
    # for i, config in tqdm(enumerate(configs)):
    #     print('[INFO] Processing #', i)
    #     graph_attack_aio(config, trace_info, output_path='./results_attack/graph_attack/ascad_v1/' + config['_id'])
