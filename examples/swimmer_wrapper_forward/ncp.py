from virtual_nematode.trainers.ncp import train_eval_test


def fully_connected():
    """ results
    units = 100
        memory: 7G * 4
        time: 2h / 300 epochs
    units = 128
        memory: 10.8G * 4
        time: 2h 45min / 300 epochs
    """
    train_eval_test(
        data_name='ncp.pt', model_name='fully_connected', lengths=[48000, 12000, 12000], batch_size=1024,
        seed=11, cuda=0, device_ids=[0, 1, 2, 3], lr=0.001, epochs=300, early_stop=30, comment='',
        # model kwargs
        units=128, output_dim=96, in_features=49
    )


def ncp():
    """ results
    memory: 7.5G * 2
    """
    train_eval_test(
        data_name='ncp.pt', model_name='ncp', lengths=[48000, 12000, 12000], batch_size=256,
        seed=11, cuda=0, device_ids=[0, 1], lr=0.001, epochs=300, early_stop=30, comment='',
        # model kwargs
        in_features=49, inter_neurons=24, command_neurons=24, motor_neurons=96, sensory_fanout=24, inter_fanout=24,
        recurrent_command_synapses=24, motor_fanin=24
    )


if __name__ == '__main__':
    fully_connected()
    # ncp()
