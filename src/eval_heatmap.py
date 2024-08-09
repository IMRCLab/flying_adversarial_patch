import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



if __name__ == '__main__':
    np.random.seed(2562)
    path = Path('eval/heatmap/')
    file_paths_gt = np.array(list(path.glob('gt/[0-9]*/anytime_losses.npy')))
    file_paths_diff = np.array(list(path.glob('diffusion/[0-9]*/anytime_losses.npy')))
    
    anytime_losses_gt = np.array([np.load(file_path) for file_path in file_paths_gt[:]])
    losses_gt = anytime_losses_gt[:, -1, 1] # select test loss of last epoch for all cells in grid

    step_size= 0.1
    y_vals = np.arange(-1, 1 + step_size, step=step_size)
    z_vals = np.arange(-0.5, 0.5 + step_size, step=step_size)

    losses_gt = np.reshape(losses_gt, (len(z_vals), len(y_vals)))




    # change settings to match latex
    plt.rcParams.update({
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.sans-serif": "Helvetica",
                "font.size": 12,
                "figure.figsize": (5, 3),
                "mathtext.fontset": 'stix'
    })

    fig, axs = plt.subplots(1, 1, layout='constrained')
    cax = axs.imshow(losses_gt, cmap='plasma', extent=[y_vals[0],y_vals[-1],z_vals[-1],z_vals[0]])
    axs.set_xlabel('target y')
    axs.set_ylabel('target z')
    axs.invert_yaxis()
    fig.colorbar(cax, orientation='vertical', shrink=0.6)
    # axs.plot(mean_timestamps_gt, mean_losses_gt, label='FAP')
    # axs.fill_between(mean_timestamps_gt, mean_losses_gt+error_losses_gt, mean_losses_gt-error_losses_gt, alpha=0.15)
    
    # axs.plot(mean_timestamps_diff, mean_losses_diff, label='diffusion')
    # axs.fill_between(mean_timestamps_diff, mean_losses_diff+error_losses_diff, mean_losses_diff-error_losses_diff, alpha=0.15)
    
    # axs.grid()
    # axs.legend()
    # axs.set_ylabel('Test loss [m]')
    # axs.set_xlabel('time in s')

    fig.savefig(f'eval/eval_heatmap.pdf', dpi=200)