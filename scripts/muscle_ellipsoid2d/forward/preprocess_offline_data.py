from virtual_nematode.data.utils import preprocess_single_dataset


if __name__ == '__main__':
    preprocess_single_dataset(load_name='data_100_2500.pt', save_name='data_100_2500_processed.pt', data_size=100, x_index=(4, 28))
