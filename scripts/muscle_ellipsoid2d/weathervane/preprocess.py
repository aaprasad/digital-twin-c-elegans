from virtual_nematode.data.utils import preprocess_and_split_dataset, preprocess_and_split_slide_dataset


if __name__ == '__main__':
    # preprocess_and_split_dataset(
    #     load_name='data_7000_640.pt',
    #     save_names=['data_7000_5000_640_64_train.pt', 'data_7000_1000_640_64_eval.pt', 'data_7000_1000_640_64_test.pt'],
    #     lengths=[5000, 1000, 1000], x_index=[(4, 28), (63, 64)], seq_len=64, seed=33
    # )
    # preprocess_and_split_slide_dataset(
    #     load_name='data_7000_640.pt',
    #     save_names=[
    #         'data_7000_5000_640_64_stride4_n10_train.pt',
    #         'data_7000_1000_640_64_stride4_n10_eval.pt',
    #         'data_7000_1000_640_64_stride4_n10_test.pt'
    #     ],
    #     lengths=[5000, 1000, 1000], x_index=[(4, 28), (63, 64)],
    #     seq_len=64, seq_amount=10, stride=4, seed=33
    # )
    # preprocess_and_split_slide_dataset(
    #     load_name='data_7000_640.pt',
    #     save_names=[
    #         'data_7000_5000_640_64_stride8_n10_train.pt',
    #         'data_7000_1000_640_64_stride8_n10_eval.pt',
    #         'data_7000_1000_640_64_stride8_n10_test.pt'
    #     ],
    #     lengths=[5000, 1000, 1000], x_index=[(4, 28), (63, 64)],  # proprioception, gradient
    #     seq_len=64, seq_amount=10, stride=8, seed=33
    # )
    preprocess_and_split_dataset(
        load_name='data_7000_640_delay100.pt',
        save_names=[
            'data_7000_5000_640_64_delay100_train.pt',
            'data_7000_1000_640_64_delay100_eval.pt',
            'data_7000_1000_640_64_delay100_test.pt'
        ],
        lengths=[5000, 1000, 1000], x_index=[(4, 28), (62, 63)],  # proprioception, concentration
        seq_len=64, seed=33
    )
