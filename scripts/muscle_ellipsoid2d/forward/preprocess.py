from virtual_nematode.data.utils import preprocess_and_split_dataset


if __name__ == '__main__':
    preprocess_and_split_dataset(
        load_name='data_7000_640.pt',
        save_names=['data_7000_5000_640_64_train.pt', 'data_7000_1000_640_64_eval.pt', 'data_7000_1000_640_64_test.pt'],
        x_range=(4, 28), lengths=[5000, 1000, 1000], seq_len=64, seed=33
    )
    """
    preprocess_and_split_dataset(
        load_name='data_3500_1280.pt',
        save_names=['data_3500_2500_1280_64_train.pt', 'data_3500_500_1280_64_eval.pt', 'data_3500_500_1280_64_test.pt'],
        lengths=[2500, 500, 500], seq_len=64, seed=33
    )
    """
