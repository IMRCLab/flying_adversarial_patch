import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



if __name__ == '__main__':
    np.random.seed(2562)
    path = Path('eval/yolov5/anytime_loss/')
    file_paths_gt = np.array(list(path.glob('gt/[0-9]*/anytime_losses.npy')))

    file_paths_diff = np.array(list(path.glob('diffusion/[0-9]*/anytime_losses.npy')))
    print(len(file_paths_diff), len(file_paths_gt))

    # random_idx = np.random.choice(len(file_paths_gt), size=100, replace=False)
    
    # ground-truth FAP anytime losses
    anytime_losses_gt = np.array([np.load(file_path) for file_path in file_paths_gt[:]])
    # print(anytime_losses_gt.shape)

    all_timestamps_gt = anytime_losses_gt[:, :, 0]   # all 102 timestamps for all 100 runs
    # print(all_timestamps_gt.shape, all_timestamps_gt[:, ::2].shape)
    all_timestamps_gt = all_timestamps_gt[:, ::2] # only every second timestamp
    mean_timestamps_gt = np.mean(all_timestamps_gt, axis=0) # mean over all 100 runs -> 102 timestamps
    
    # print(mean_timestamps_gt[:10])
    # log_meantime_gt = np.log10(mean_timestamps_gt)

    all_losses_gt = anytime_losses_gt[:, :, 1] # all 102 losses for all 100 runs
    all_losses_gt = all_losses_gt[:, ::2] # only every second loss
    mean_losses_gt = np.mean(all_losses_gt, axis=0) # mean over all 100 runs -> 102 losses
    error_losses_gt = np.std(all_losses_gt, axis=0) # std over all 100 runs -> 102 entries

    # diffusion anytime losses
    anytime_losses_diff = np.array([np.load(file_path) for file_path in file_paths_diff[:]])

    all_timestamps_diff = anytime_losses_diff[:, :, 0]   # all 102 timestamps for all 100 runs
    all_timestamps_diff = all_timestamps_diff[:, ::2] # only every second timestamp
    mean_timestamps_diff = np.mean(all_timestamps_diff, axis=0) # mean over all 100 runs -> 102 timestamps
    # print(mean_timestamps_diff[:10])
    # log_meantime_diff = np.log10(mean_timestamps_diff)

    all_losses_diff = anytime_losses_diff[:, :, 1] # all 102 losses for all 100 runs
    all_losses_diff = all_losses_diff[:, ::2] # only every second loss
    mean_losses_diff = np.mean(all_losses_diff, axis=0) # mean over all 100 runs -> 102 losses
    error_losses_diff = np.std(all_losses_diff, axis=0) # std over all 100 runs -> 102 entries

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
    axs.plot(mean_timestamps_gt, mean_losses_gt, label='from random')
    axs.fill_between(mean_timestamps_gt, mean_losses_gt+error_losses_gt, mean_losses_gt-error_losses_gt, alpha=0.15)
    
    axs.plot(mean_timestamps_diff, mean_losses_diff, label='from diffusion')
    axs.fill_between(mean_timestamps_diff, mean_losses_diff+error_losses_diff, mean_losses_diff-error_losses_diff, alpha=0.15)
    
    # for idx in range(2, 7):
    axs.axvline(x=mean_timestamps_gt[4], alpha=0.4, linestyle='--')
    axs.axvline(x=mean_timestamps_diff[4], alpha=0.4, color='#ff7f0e', linestyle='--')

    axs.grid()
    axs.legend()
    axs.set_ylabel('Test loss [m]')
    axs.set_xlabel('time in s')
    axs.set_xscale('log')
    axs.set_yscale('log')
    # axs.set_xlim(left=0, right=10^2)

    fig.savefig(f'eval/yolov5/eval_anytime_loss_3e.pdf', dpi=200)