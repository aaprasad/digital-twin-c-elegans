from matplotlib import pyplot as plt
import seaborn as sns
from virtual_nematode.connectomes.weathervane import get_kwargs
import worm_assets


def weathervane_coefficient():
    x = [
        0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2,
        2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0
    ]
    y = [
        0.14, 0.17, 0.25, 0.35, 0.39, 0.43, 0.45, 0.51, 0.57, 0.62, 0.66, 0.69, 0.70, 0.72, 0.73, 0.73, 0.74, 0.74,
        0.75, 0.75, 0.76, 0.76, 0.77, 0.77, 0.77, 0.78, 0.78, 0.79, 0.79, 0.79, 0.79, 0.80, 0.80, 0.80, 0.80, 0.80,
        0.80, 0.80, 0.80, 0.81, 0.81
    ]
    y_start = [0.37] * len(y)
    # plot
    plt.plot(x, y, label='chemotaxis index mean')
    plt.plot(x, y_start, label='start concentration mean')
    plt.axvline(x=1.1, ls='-')
    plt.legend()
    plt.xlabel('k_w')
    plt.ylabel('chemotaxis index mean')
    print(x[10], y[10])
    print(x[11], y[11])
    print(x[12], y[12])


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


if __name__ == '__main__':
    runs_folder = 'Jul25_12-07-58_h-10-176-50-34'
    ckpt_name = 'model119.pt'
    model_name = 'snn_weathervane'
    kwargs = get_kwargs(
        path=worm_assets.connectome_path(),
        polarity_path=worm_assets.polarity_path('Cook et al connectome.xls')
    )
