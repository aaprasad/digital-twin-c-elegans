""" train NCP swimmer of chemotaxis behavior with offline testing """

from virtual_nematode.trainers.ncp import train_eval_test


if __name__ == '__main__':
    train_eval_test(
        cuda=1, batch_size=2048, epochs=100, data_name='ncp.pt', model_name='fully_connected',
        lengths=[65600, 14000, 14000]
    )
