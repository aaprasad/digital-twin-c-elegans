from eval import select_model
from matplotlib import pyplot as plt
import os
import seaborn as sns
import torch


def plot_model_weight(model, ckpt_name):
    plt.figure(figsize=(20, 10))
    # chemical weight
    w_c = (
            model.cell.w_c.abs() * model.cell.w_c_ex_mask -
            model.cell.w_c.abs() * model.cell.w_c_in_mask +
            model.cell.w_c * (
                    model.cell.w_c_mask.float() -
                    model.cell.w_c_ex_mask.float() -
                    model.cell.w_c_in_mask.float()
        )
    )
    w_c = w_c.clone().detach()
    w_c_max = w_c.abs().max().item()
    plt.subplot(2, 3, 1)
    plt.title('chemical weight')
    sns.heatmap(w_c, cmap='coolwarm', vmin=-w_c_max, vmax=w_c_max)
    # proprioception input weight
    w_p = model.cell.w_p * model.cell.w_p_mask
    w_p = w_p.clone().detach()
    w_p_max = w_p.abs().max().item()
    plt.subplot(2, 3, 2)
    plt.title('proprioception input weight')
    sns.heatmap(w_p, cmap='coolwarm', vmin=-w_p_max, vmax=w_p_max)
    # tau
    tau = model.cell.tau.clamp(0.01, 0.05)
    tau = tau.clone().detach()
    plt.subplot(2, 3, 3)
    plt.title('tau')
    plt.plot(tau)
    # gap junction weight
    w_g = model.cell.w_g.abs()
    w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * model.cell.w_g_mask
    w_g = w_g.clone().detach()
    w_g_max = w_g.max().item()
    plt.subplot(2, 3, 4)
    plt.title('gap junction weight')
    sns.heatmap(w_g, cmap='coolwarm', vmin=0, vmax=w_g_max)
    # output scaling weight
    w_output = model.cell.w_output.clamp(0, 1)
    w_output = w_output.clone().detach()
    plt.subplot(2, 3, 5)
    plt.title('output scaling weight')
    plt.plot(w_output)
    # bias
    bias = model.cell.bias.clone().detach()
    plt.subplot(2, 3, 6)
    plt.title('bias')
    plt.plot(bias)
    plt.savefig(os.path.join(data_path, '{}.png'.format(ckpt_name)))


def plot_action(data_name, ckpt_name):
    dataset = torch.load(os.path.join('data', data_name), map_location=torch.device('cpu'))
    x, y = dataset.tensors
    print(x.shape, y.shape)
    ckpt_path = os.path.join(data_path, ckpt_name)
    action = torch.load(ckpt_path, map_location=torch.device('cpu'))
    print(action.shape)
    name = ['dl', 'dr', 'vl', 'vr']
    for i in range(24):
        plt.figure(i, figsize=(10, 10))
        for j, start in enumerate([0, 24, 48, 71]):
            if name[j] == 'vl' and i == 23:
                continue
            plt.subplot(2, 2, j + 1)
            index = start + i
            plt.plot(y[0, 0:640, index], label='data')
            plt.plot(action[0:640, index], label='nn')
            plt.title(name[j] + str(i + 1) + '-' + str(index))
        plt.legend()
        plt.savefig(ckpt_path + '.{:02d}.png'.format(i + 1))
        plt.close(i)


def plot_action_heatmap(ckpt_name):
    ckpt_path = os.path.join(data_path, ckpt_name)
    action = torch.load(ckpt_path, map_location=torch.device('cpu'))
    print(action.shape)
    plt.figure(figsize=(13, 10))
    sns.heatmap(action[0:320, :].clamp(0, 1).T, cmap='coolwarm')
    plt.savefig(ckpt_path + '.png')


if __name__ == '__main__':
    runs_folder = ''
    ckpt_name = 'model.pt'
    model_name = 'snn_forward'
    model_folder = os.path.join('runs', runs_folder)
    data_path = os.path.join('data', runs_folder)
    model = select_model(model_folder, model_name, ckpt_name)
    plot_model_weight(model, ckpt_name)
    # plot_action('data_new_640.pt', ckpt_name)
    plot_action_heatmap(ckpt_name)
