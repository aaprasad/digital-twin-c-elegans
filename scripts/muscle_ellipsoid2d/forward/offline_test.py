import os
import sys
import torch
from virtual_nematode.connectomes.forward import get_kwargs
from virtual_nematode.trainers.ncp import offline_test, offline_test_per_sample
import worm_assets


if __name__ == '__main__':
    runs_folder = sys.argv[1]
    ckpt_name = sys.argv[2]  # 'model.pt'
    model_folder = os.path.join('runs', runs_folder)
    print(model_folder, ckpt_name)
    # data_name = 'data_7000_1000_640_64_eval.pt'  # 'data_7000_1000_640_64_test.pt'
    data_name = 'data_7000_1000_640_64_stride8_n10_eval.pt'
    kwargs = {
        'dt': 0.04, 'steps': 5,
        **get_kwargs(
            path=worm_assets.connectome_path(),
            polarity_path=worm_assets.polarity_path('Cook et al connectome.xls')
        )
    }
    # mse = offline_test(data_name, 'li_conductance', model_folder, ckpt_name, batch_size=128, device_ids=0, **kwargs)
    mse = offline_test_per_sample(data_name, 'li_conductance', model_folder, ckpt_name, batch_size=128, device_ids=0, **kwargs)
    # print('{:.4e}'.format(mse))
    print(mse.shape, '{:.4e}'.format(mse.mean().item()))
    save_path = os.path.join('data', runs_folder, ckpt_name)
    os.makedirs(save_path, exist_ok=True)
    torch.save(mse, os.path.join(save_path, 'eval_loss.pt'))
