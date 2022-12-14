import copy
import os
import sys
import torch
from virtual_nematode.data.utils import prepare_test_dataloader
from virtual_nematode.trainers.ncp import test_none_reduction
from test import select_model
from test_ablation_once import (
    modify_all_mask, modify_chemical_mask, modify_chemical_input_mask, modify_chemical_output_mask,
    modify_electrical_mask
)


def offline_test(data_name, model_name, model_folder, ckpt_name, device_id):
    # data
    data_path = os.path.join('data', data_name)
    test_loader = prepare_test_dataloader(data_path, batch_size=128)
    # model
    device = torch.device('cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu')
    model = select_model(model_folder, model_name, ckpt_name)
    model = model.to(device)
    criterion = torch.nn.MSELoss(reduction='none')
    # offline test
    mse = test_none_reduction(model, device, test_loader, criterion)
    mse = mse.detach()
    # result
    torch.save(mse, os.path.join(save_folder, 'test_loss_2500.control.pt'))
    print('MSE Loss', mse.shape, '{:.4e}'.format(mse.mean().item()))


def offline_test_single_ablation(data_name, model_name, model_folder, ckpt_name, device_id):
    # data
    data_path = os.path.join('data', data_name)
    test_loader = prepare_test_dataloader(data_path, batch_size=128)
    # model
    device = torch.device('cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu')
    model = select_model(model_folder, model_name, ckpt_name)
    criterion = torch.nn.MSELoss(reduction='none')
    # save path
    save_folder_temp = os.path.join(save_folder, 'offline_test_all_2500')
    # save_folder_temp = os.path.join(save_folder, 'offline_test_chemical_2500')
    # save_folder_temp = os.path.join(save_folder, 'offline_test_chemical_input_2500')
    # save_folder_temp = os.path.join(save_folder, 'offline_test_chemical_output_2500')
    # save_folder_temp = os.path.join(save_folder, 'offline_test_electrical_2500')
    os.makedirs(save_folder_temp, exist_ok=True)
    for i in range(469):
        model_temp = copy.deepcopy(model)
        # ablation
        model_temp = modify_all_mask(model_temp, index=i)
        # model_temp = modify_chemical_mask(model_temp, index=i)
        # model_temp = modify_chemical_input_mask(model_temp, index=i)
        # model_temp = modify_chemical_output_mask(model_temp, index=i)
        # model_temp = modify_electrical_mask(model_temp, index=i)
        # offline test
        model_temp = model_temp.to(device)
        mse = test_none_reduction(model_temp, device, test_loader, criterion)
        mse = mse.detach()
        # result
        torch.save(mse, os.path.join(save_folder_temp, 'test_loss.{}.pt'.format(i)))
        print('MSE Loss', mse.shape, '{:.4e}'.format(mse.mean().item()))


if __name__ == '__main__':
    runs_folder = sys.argv[1]
    ckpt_name = sys.argv[2]  # 'model.pt'
    model_folder = os.path.join('runs', runs_folder)
    save_folder = os.path.join('data', runs_folder, ckpt_name)
    # data_name = 'data_7000_1000_640_64_stride8_n10_test.pt'
    data_name = 'data_100_2500_processed.pt'  # 100 trials, 2500 steps
    print(model_folder, ckpt_name, data_name)
    os.makedirs(save_folder, exist_ok=True)
    model_name = 'li_conductance'
    # offline_test(data_name, model_name, model_folder, ckpt_name, device_id=0)
    offline_test_single_ablation(data_name, model_name, model_folder, ckpt_name, device_id=0)
