import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



if __name__ == '__main__':
    np.random.seed(2562)
    path = Path('eval/heatmap/')
    file_paths_gt = np.array(list(path.glob('gt/[0-9]*/anytime_losses.npy')))
    file_paths_diff = np.array(list(path.glob('diffusion/[0-9]*/anytime_losses.npy')))
    
    random_idx = np.random.choice(len(file_paths_gt), size=100, replace=False)
    
    # ground-truth FAP anytime losses
    anytime_losses_gt = np.array([np.load(file_path) for file_path in file_paths_gt[:]])

    all_timestamps_gt = anytime_losses_gt[:, :, 0]   # all 102 timestamps for all 100 runs
    mean_timestamps_gt = np.mean(all_timestamps_gt, axis=0)[:20] # mean over all 100 runs -> 102 timestamps

    all_losses_gt = anytime_losses_gt[:, :, 1] # all 102 losses for all 100 runs
    mean_losses_gt = np.mean(all_losses_gt, axis=0)[:20] # mean over all 100 runs -> 102 losses
    error_losses_gt = np.std(all_losses_gt, axis=0)[:20] # std over all 100 runs -> 102 entries

    # diffusion anytime losses
    anytime_losses_diff = np.array([np.load(file_path) for file_path in file_paths_diff])

    all_timestamps_diff = anytime_losses_diff[:, :, 0]   # all 102 timestamps for all 100 runs
    mean_timestamps_diff = np.mean(all_timestamps_diff, axis=0)[:20] # mean over all 100 runs -> 102 timestamps

    all_losses_diff = anytime_losses_diff[:, :, 1] # all 102 losses for all 100 runs
    mean_losses_diff = np.mean(all_losses_diff, axis=0)[:20] # mean over all 100 runs -> 102 losses
    error_losses_diff = np.std(all_losses_diff, axis=0)[:20] # std over all 100 runs -> 102 entries

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
    axs.plot(mean_timestamps_gt, mean_losses_gt, label='FAP from random')
    axs.fill_between(mean_timestamps_gt, mean_losses_gt+error_losses_gt, mean_losses_gt-error_losses_gt, alpha=0.15)
    
    axs.plot(mean_timestamps_diff, mean_losses_diff, label='FAP from diffusion')
    axs.fill_between(mean_timestamps_diff, mean_losses_diff+error_losses_diff, mean_losses_diff-error_losses_diff, alpha=0.15)
    
    axs.grid()
    axs.legend()
    axs.set_ylabel('Test loss [m]')
    axs.set_xlabel('time [s]')

    fig.savefig(f'eval/eval_anytime_loss.pdf', dpi=200)