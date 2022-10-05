"""
In this script, we show how to perform side-channel characterisation using Welch's T-test to study the behaviour of an Aes Sbox

The characterisation is made with the TTestEngine
Its constructor needs a partition function, which will separate leakages into two classes.

"""
from Lib_SCA.lascar import Session, MatPlotLibOutputMethod, SimulatedPowerTraceFixedRandomContainer, TTestEngine


def t_test(config_name, engine_name='ttest', batch_size=2500):
    container = SimulatedPowerTraceFixedRandomContainer(config_name)
    a_byte = int(input('pls choose one byte from ' + str(container.idx_exp) + ': '))
    c = container.fixed_set[a_byte]

    def partition_function(
            value, attack_byte=a_byte, attack_time=container.attack_sample_point, cipher=c
    ):  # partition_function must take 1 argument: the value returned by the container at each trace
        return int(value["ciphertext"][attack_byte][attack_time] == cipher)

    ttest_engine = TTestEngine(engine_name, partition_function)

    # We choose here to plot the resulting curve
    plot_output = MatPlotLibOutputMethod(ttest_engine)
    session = Session(container, output_method=plot_output)
    session.add_engine(ttest_engine)

    session.run(batch_size=batch_size)


if __name__ == '__main__':
    t_test(config_name='fixed_random_traces.yaml', engine_name='ttest', batch_size=2500)
