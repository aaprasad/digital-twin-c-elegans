import csv
from test import select_model
from matplotlib import pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch
from virtual_nematode.connectomes.cells import body_wall_muscles, cell_list
from virtual_nematode.connectomes.forward import get_kwargs
import worm_assets


def plot_mask(**kwargs):
    # masks
    w_c_mask = kwargs.get('w_c_mask')
    w_g_mask = kwargs.get('w_g_mask')
    w_p_mask = kwargs.get('w_p_mask')
    # chemical
    plt.figure()
    plt.title('Exc/Inh Chemical Synapse Mask {}'.format(w_c_mask[0].sum().item()))
    sns.heatmap(w_c_mask[0], cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('Cell ID')
    plt.ylabel('Cell ID')
    plt.figure()
    plt.title('Excitatory Chemical Synapse Mask {}'.format(w_c_mask[1].sum().item()))
    sns.heatmap(w_c_mask[1], cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('Cell ID')
    plt.ylabel('Cell ID')
    plt.figure()
    plt.title('Inhibitory Chemical Synapse Mask {}'.format(w_c_mask[2].sum().item()))
    sns.heatmap(w_c_mask[2], cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('Cell ID')
    plt.ylabel('Cell ID')
    # gap junction
    plt.figure()
    plt.title('Gap Junction Mask {}'.format(w_g_mask.sum().item()))
    sns.heatmap(w_g_mask, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('Cell ID')
    plt.ylabel('Cell ID')
    # proprioception
    plt.figure()
    plt.title('Exc/Inh Proprioception Input Mask')
    sns.heatmap(w_p_mask[0], cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('Cell ID')
    plt.ylabel('Joint ID')
    plt.figure()
    plt.title('Excitatory Proprioception Input Mask')
    sns.heatmap(w_p_mask[1], cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('Cell ID')
    plt.ylabel('Joint ID')
    plt.figure()
    plt.title('Inhibitory Proprioception Input Mask')
    sns.heatmap(w_p_mask[2], cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('Cell ID')
    plt.ylabel('Joint ID')


def plot_weight(runs_folder, ckpt_name, model_name):
    model = select_model(os.path.join('runs', runs_folder), model_name, ckpt_name)
    # plot
    plt.figure(figsize=(20, 10))
    w_c = model.cell.w_c.clone().detach()
    w_c_mask = model.cell.w_c_mask.clone().detach()
    w_c = w_c * w_c_mask
    print('w_c: min {}, max {}, mean {}, std {}'.format(w_c.min().item(), w_c.max().item(), w_c.mean().item(), w_c.std().item()))
    w_c_max = w_c.std().item()  # w_c.abs().max().item()
    plt.subplot(2, 3, 1)
    plt.title('Chemical Synapse Weight')
    sns.heatmap(w_c, cmap='coolwarm', vmin=-w_c_max, vmax=w_c_max)
    plt.xlabel('Cell ID')
    plt.ylabel('Cell ID')
    # gap junction weight
    w_g = model.cell.w_g.clone().detach()
    w_g_mask = model.cell.w_g_mask.clone().detach()
    w_g = w_g.abs()
    w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * w_g_mask
    print('w_g: min {}, max {}, mean {}, std {}'.format(w_g.min().item(), w_g.max().item(), w_g.mean().item(), w_g.std().item()))
    w_g_max = w_g.std().item()  # w_g.max().item()
    plt.subplot(2, 3, 2)
    plt.title('Gap Junction Weight')
    sns.heatmap(w_g, cmap='coolwarm', vmin=-w_g_max, vmax=w_g_max)
    plt.xlabel('Cell ID')
    plt.ylabel('Cell ID')
    # proprioception input weight
    w_p = model.cell.w_p
    w_p_mask = model.cell.w_p_mask
    w_p = w_p * w_p_mask[0] + w_p.abs() * w_p_mask[1] - w_p.abs() * w_p_mask[2]
    print('w_p: min {}, max {}, mean {}, std {}'.format(w_p.min().item(), w_p.max().item(), w_p.mean().item(), w_p.std().item()))
    w_p_max = w_p.std().item()  # w_p.abs().max().item()
    plt.subplot(2, 3, 3)
    plt.title('Proprioception Input Weight')
    sns.heatmap(w_p, cmap='coolwarm', vmin=-w_p_max, vmax=w_p_max)
    plt.xlabel('Cell ID')
    plt.ylabel('Joint ID')
    # tau
    tau = model.cell.tau.clone().detach()
    tau = tau.clamp(0.01, 0.05)
    print('tau: min {}, max {}, mean {}, std {}'.format(tau.min().item(), tau.max().item(), tau.mean().item(), tau.std().item()))
    plt.subplot(2, 3, 4)
    plt.title('tau')
    plt.plot(tau)
    # bias
    bias = model.cell.bias.clone().detach()
    print('bias: min {}, max {}, mean {}, std {}'.format(bias.min().item(), bias.max().item(), bias.mean().item(), bias.std().item()))
    plt.subplot(2, 3, 5)
    plt.title('Bias')
    plt.plot(bias)
    plt.xlabel('Cell ID')
    plt.ylabel('Bias')


def chemical_weight_polarity(runs_folder, ckpt_name, model_name):
    model = select_model(os.path.join('runs', runs_folder), model_name, ckpt_name)
    w_c = model.cell.w_c * model.cell.w_c_mask
    w_c = w_c.clone().detach()
    print('w_c', w_c.shape)
    print('w_c: min {}, max {}, mean {}, std {}'.format(w_c.min().item(), w_c.max().item(), w_c.mean().item(), w_c.std().item()))
    inhibitory = torch.sum(w_c < 0).item()
    excitatory = torch.sum(w_c > 0).item()
    print('< 0: ', inhibitory, inhibitory / (inhibitory + excitatory))
    print('> 0: ', excitatory, excitatory / (inhibitory + excitatory))
    print('= 0: ', torch.sum(w_c == 0).item())


def chemical_weight_density(runs_folder, ckpt_name, model_name):
    model = select_model(os.path.join('runs', runs_folder), model_name, ckpt_name)
    w_c = model.cell.w_c * model.cell.w_c_mask
    w_c = w_c.clone().detach()
    print('w_c', w_c.shape)
    w_c = w_c[w_c != 0]  # remove no connection
    print('w_c', w_c.shape)
    w_c = w_c ** 2
    w_c = w_c[w_c < 10]
    print('w_c', w_c.shape)
    plt.hist(w_c, bins=10, range=None, density=True)


def plot_output(runs_folder, ckpt_name, steps):
    x, y = torch.load(os.path.join('data', runs_folder, ckpt_name), map_location=torch.device('cpu'))
    print(y.shape)
    n = 469
    state = y[:, 0:n]
    activation = y[:, n:n+n]
    action = y[:, n+n:]
    # plot
    plt.figure()
    plt.title('State')
    sns.heatmap(state[0:steps, :].clamp(0, 1).T, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('t (step)')
    plt.ylabel('Cell ID')
    plt.figure()
    plt.title('Activation')
    sns.heatmap(activation[0:steps, :].clamp(0, 1).T, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('t (step)')
    plt.ylabel('Cell ID')
    plt.figure()
    plt.title('Action')
    sns.heatmap(action[0:steps, :].clamp(0, 1).T, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('t (step)')
    plt.ylabel('Muscle ID')
    return state, activation, action


def plot_state_range(runs_folder, ckpt_name):
    # state
    ckpt_path = os.path.join('data', runs_folder, ckpt_name[0:-3] + '.state.pt')
    state = torch.load(ckpt_path, map_location=torch.device('cpu'))
    # state = state.squeeze(dim=1)
    print(state.shape)
    # cells
    cells = cell_list(path=worm_assets.connectome_path())
    # max and min
    plt.figure(figsize=(10, 10))
    plt.title('Stats about Cell State Time Sequence')
    plt.plot(state.max(dim=0).values, label='Max')
    plt.plot(state.min(dim=0).values, label='Min')
    plt.legend()
    plt.xlabel('Cell ID')
    plt.ylabel('value')
    plt.savefig(ckpt_path + '1.png')
    plt.close()
    # median
    plt.figure(figsize=(10, 10))
    plt.title('Stats about Cell State Time Sequence')
    plt.plot(state.median(dim=0).values, label='Median')
    plt.legend()
    plt.xlabel('Cell ID')
    plt.ylabel('value')
    plt.savefig(ckpt_path + '2.png')
    plt.close()
    # mean
    plt.figure(figsize=(10, 10))
    plt.title('Stats about Cell State Time Sequence')
    state_mean = state.mean(dim=0)
    plt.plot(state_mean, label='Mean')
    plt.legend()
    plt.xlabel('Cell ID')
    plt.ylabel('value')
    plt.savefig(ckpt_path + '3.png')
    plt.close()
    # csv: mean
    with open(os.path.join('data', runs_folder, '{}.cell_state_mean.csv'.format(ckpt_name)), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(('Cell ID', 'Cell Name', 'Cell State Mean'))
        for i, name in enumerate(cells):
            writer.writerow((i, name, state_mean[i].item()))


def weight_analysis(runs_folder, ckpt_name, model_name, remove_muscle_flag=True):
    model = select_model(os.path.join('runs', runs_folder), model_name, ckpt_name)
    # cells
    muscles = body_wall_muscles()
    cells = cell_list(path=worm_assets.connectome_path())
    # chemical synapse weight
    w_c = model.cell.w_c * model.cell.w_c_mask
    w_c = w_c.clone().detach()
    w_c_mask = model.cell.w_c_mask.clone().detach()
    # gap junction weight
    w_g = model.cell.w_g.abs()
    w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * model.cell.w_g_mask
    w_g = w_g.clone().detach()
    w_g_mask = model.cell.w_g_mask.clone().detach()
    # proprioception input weight
    w_p = model.cell.w_p * model.cell.w_p_mask
    w_p = w_p.clone().detach()
    w_p_mask = model.cell.w_p_mask.clone().detach()
    # analysis: chemical input
    chemical_input_number = torch.sum(w_c_mask, dim=0)
    chemical_input_number[chemical_input_number == 0] = 1
    chemical_input = torch.sum(w_c ** 2, dim=0) / chemical_input_number
    # analysis: chemical output
    chemical_output_number = torch.sum(w_c_mask, dim=1)
    chemical_output_number[chemical_output_number == 0] = 1
    chemical_output = torch.sum(w_c ** 2, dim=1) / chemical_output_number
    # analysis: chemical
    chemical_number = torch.sum(w_c_mask, dim=0) + torch.sum(w_c_mask, dim=1)
    chemical_number[chemical_number == 0] = 1
    chemical = (torch.sum(w_c ** 2, dim=0) + torch.sum(w_c ** 2, dim=1)) / chemical_number
    # analysis: gap junction
    gap_number = torch.sum(w_g_mask, dim=0)
    gap_number[gap_number == 0] = 1
    gap = torch.sum(w_g ** 2, dim=0) / gap_number
    # analysis: proprioception input
    proprioception_input_number = torch.sum(w_p_mask, dim=0)
    proprioception_input_number[proprioception_input_number == 0] = 1
    proprioception_input = torch.sum(w_p ** 2, dim=0) / proprioception_input_number
    # csv
    with open(os.path.join('data', runs_folder, '{}.weight_analysis.csv'.format(ckpt_name)), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(('Cell ID', 'Cell Name', 'Chemical Input', 'Chemical Output', 'Chemical', 'Gap Junction', 'Proprioception Input'))
        for i, name in enumerate(cells):
            if remove_muscle_flag is True and name in muscles:
                continue
            writer.writerow((i, name, chemical_input[i].item(), chemical_output[i].item(), chemical[i].item(), gap[i].item(), proprioception_input[i].item()))


def bias(runs_folder, ckpt_name, model_name, remove_muscle_flag=True):
    model = select_model(os.path.join('runs', runs_folder), model_name, ckpt_name)
    # cells
    muscles = body_wall_muscles()
    cells = cell_list(path=worm_assets.connectome_path())
    # bias
    bias = model.cell.bias.clone().detach()
    with open(os.path.join('data', runs_folder, '{}.bias.csv'.format(ckpt_name)), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(('Cell ID', 'Cell Name', 'Bias'))
        for i, name in enumerate(cells):
            if remove_muscle_flag is True and name in muscles:
                continue
            writer.writerow((i, name, bias[i].item()))


def plot_loss(path):
    # plot loss by muscle
    data = torch.load(path)
    mse = data.mean(dim=0)
    mse_mean = data.mean()
    plt.plot(mse[0:24], label='DL')
    plt.plot(mse[24:48], label='DR')
    plt.plot(mse[48:71], label='VL')
    plt.plot(mse[71:95], label='VR')
    plt.hlines(mse_mean, 0, 23, label='mean', linestyle='dashed', color='purple')
    plt.legend()
    plt.xlabel('Joint ID')
    plt.ylabel('MSE')
    plt.title('MSE by Muscle')
    # muscle with loss higher than average
    index = np.arange(24)
    print('DL', index[mse[0:24] > mse_mean])
    print('DR', index[mse[24:48] > mse_mean])
    print('VL', index[0:23][mse[48:71] > mse_mean])
    print('VR', index[mse[71:95] > mse_mean])


if __name__ == '__main__':
    runs_folder = 'Jul25_12-07-58_h-10-176-50-34'
    ckpt_name = 'model119.pt'
    model_name = 'snn_forward'
    kwargs = get_kwargs(
        path=worm_assets.connectome_path(),
        polarity_path=worm_assets.polarity_path('Cook et al connectome.xls')
    )
