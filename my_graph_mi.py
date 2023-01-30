from Lib_SCA.lascar import SimulatedPowerTraceFixedRandomContainer
from Lib_SCA.lascar import Session, GraphMIEngine


def graph_based_mi(params, trace_params):
    container = None
    if params['mode'] == 'fix_random':
        container = SimulatedPowerTraceFixedRandomContainer(config_params=trace_params)
    elif params['mode'] == 'real':
        pass

    def partition_function(value):
        # partition_function must take 1 argument: the value returned by the container at each trace
        # fix and random sets have already been partitioned in container
        return int(value["trace_idx"] % 2 != 0)

    graph_test_engine = GraphMIEngine(params['engine_name'],
                                      partition_function,
                                      time_delay=2,
                                      dim=3)
    output_path = trace_params['_id']

    # We choose here to plot the resulting curve
    session = Session(container, engine=graph_test_engine)
    session.run(batch_size=params['batch_size'])
