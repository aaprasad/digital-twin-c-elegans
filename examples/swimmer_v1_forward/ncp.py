from virtual_nematode.trainers.ncp import train_eval_test


if __name__ == '__main__':
    """ results
    memory: 7597MiB
    time: ~ 1h 50min / 300 epochs
    """
    train_eval_test(
        data_name='ncp.pt', model_name='fully_connected', lengths=[48000, 12000, 12000], batch_size=512,
        seed=42, cuda=0, device_ids=[0, 1, 2, 3], lr=0.001, epochs=300, early_stop=30, comment='',
        # model kwargs
        units=100, output_dim=24, in_features=49
    )
