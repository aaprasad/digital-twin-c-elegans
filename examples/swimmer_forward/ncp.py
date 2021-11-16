from virtual_nematode.trainers.ncp import train_eval_test


def fully_connected():
    """ results
    units = 100
        memory: 7G * 4
        time: ~ 2h / 300 epochs
    units = 72
        memory: 4.9G * 4
        time: ~ 2h / 300 epochs
    """
    train_eval_test(
        data_name='ncp.pt', model_name='fully_connected', lengths=[48000, 12000, 12000], batch_size=1024,
        seed=11, cuda=0, device_ids=[0, 1, 2, 3], lr=0.001, epochs=300, early_stop=30, comment='',
        # model kwargs
        units=150, output_dim=24, in_features=49
    )


def ncp():
    """ results
    memory: 4.9G * 4
    time: ~ 2h / 300 epochs
    """
    train_eval_test(
        data_name='ncp.pt', model_name='ncp', lengths=[48000, 12000, 12000], batch_size=1024,
        seed=11, cuda=0, device_ids=[0, 1, 2, 3], lr=0.001, epochs=300, early_stop=30, comment='',
        # model kwargs
        in_features=49, inter_neurons=12, command_neurons=36, motor_neurons=24, sensory_fanout=6, inter_fanout=12,
        recurrent_command_synapses=6, motor_fanin=6
    )


if __name__ == '__main__':
    fully_connected()
    # ncp()
