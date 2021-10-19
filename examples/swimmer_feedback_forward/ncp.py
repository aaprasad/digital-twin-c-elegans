from virtual_nematode.trainers.ncp import offline_train_and_test


if __name__ == '__main__':
    offline_train_and_test(
        data_name='ncp.pt', model_name='fully_connected', lengths=[48000, 12000, 12000], batch_size=128,
        seed=42, cuda=0, lr=0.001, epochs=300, early_stop=30, comment='',
        # model kwargs
        units=30, output_dim=11, in_features=1
    )
