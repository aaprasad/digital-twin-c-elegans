import os
import torch
from virtual_nematode.data.subset import random_split_data
from virtual_nematode.data.utils import preprocess_and_split_slide_dataset


if __name__ == '__main__':
    # load
    dataset = torch.load('data/data_35000_320.pt')
    print('load', dataset.tensors[0].shape, dataset.tensors[1].shape)
    # split
    datasets = random_split_data(dataset, lengths=[7000] * 5, seed=42)
    del dataset
    for i, d in enumerate(datasets):
        print('random split', i, d.tensors[0].shape, d.tensors[1].shape)
        save_name = 'data_7000_320.{}.pt'.format(i)
        torch.save(d, os.path.join('data', save_name))
        save_folder = 'experiment' + str(i)
        os.makedirs(os.path.join('data', save_folder), exist_ok=True)
        preprocess_and_split_slide_dataset(
            load_name=save_name,
            save_names=[
                os.path.join(save_folder, 'data_7000_320_64_stride8_n10_train.pt'),
                os.path.join(save_folder, 'data_7000_320_64_stride8_n10_eval.pt'),
                os.path.join(save_folder, 'data_7000_320_64_stride8_n10_test.pt')
            ],
            lengths=[5000, 1000, 1000], x_index=(4, 28),
            seq_len=64, seq_amount=10, stride=8, seed=33
        )
