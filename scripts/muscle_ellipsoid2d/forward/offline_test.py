import os
import sys
from virtual_nematode.connectomes.forward import get_kwargs
from virtual_nematode.trainers.ncp import offline_test
import worm_assets


if __name__ == '__main__':
    runs_folder = sys.argv[1]
    ckpt_name = sys.argv[2]
    model_folder = os.path.join('runs', runs_folder)
    print(model_folder, ckpt_name)
    data_name = 'data_7000_1000_640_64_eval.pt'
    # data_name = 'data_7000_1000_640_64_test.pt'
    kwargs = {
        'dt': 0.04, 'steps': 5,
        **get_kwargs(
            path=worm_assets.connectome_path(),
            polarity_path=worm_assets.polarity_path('Cook et al connectome.xls')
        )
    }
    offline_test(data_name, 'snn_forward3', model_folder, ckpt_name, batch_size=128, device_ids=0, **kwargs)
