import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import pickle

def calc_loss_dataset():
    # get coordinates for the patches that were included in train dataset
    with open('diffusion/FAP_combined.pickle', 'rb') as f:
        data = pickle.load(f)    

    # patches = []
    targets = []
    # positions = []
    for i in range(len(data)):
        # patches.append(data[i][0])
        targets.append(data[i][1])
        # positions.append(data[i][2])
    
    # patches = np.array(patches)
    targets = np.array(targets)
    # positions = np.array(positions)

    indices = np.argwhere((targets[:, 0] > 1.99) & (targets[:, 0] < 2.01)).flatten()
    print(indices, len(indices))

    return np.round(targets[indices, 1:], 1)

if __name__ == '__main__':
    gt_coords = calc_loss_dataset()
    print(gt_coords)

    path = Path('eval/heatmap/')
    # file_paths_gt = list(path.glob('[0-9]*/gt/[0-9]*/anytime_losses.npy'))
    # file_paths_gt.sort(key=lambda path: int(path.parent.name))

    
    # file_paths_diff = list(path.glob('[0-9]*/diffusion/[0-9]*/anytime_losses.npy'))
    # file_paths_diff.sort(key=lambda path: int(path.parent.name))

    anytime_losses_gt = []
    for i in range(10):
        file_paths_gt = list(path.glob(str(i)+'/gt/[0-9]*/anytime_losses.npy'))
        file_paths_gt.sort(key=lambda path: int(path.parent.name))
        if i == 0:
            losses = np.array([np.load(file_path) for file_path in file_paths_gt])
            # shape: 231, 102, 2 --> 231 values, 102 epochs (0 ==> np.inf), time+loss
            losses = losses[:, 5, 1] # only keep first 10 epochs
            # losses = losses[:, -1]
        else:
            losses = np.array([np.load(file_path) for file_path in file_paths_gt])
            losses = losses[:, 5, 1] 

        anytime_losses_gt.append(losses)

    anytime_losses_gt = np.array(anytime_losses_gt)
    print(anytime_losses_gt.shape)
    
    anytime_losses_diff = []
    for i in range(10):
        file_paths_gt = list(path.glob(str(i)+'/diffusion/[0-9]*/anytime_losses.npy'))
        file_paths_gt.sort(key=lambda path: int(path.parent.name))
        # if i == 0:
        #     losses = np.array([np.load(file_path) for file_path in file_paths_gt])
        #     # shape: 231, 102, 2 --> 231 values, 102 epochs (0 ==> np.inf), time+loss
        #     losses = losses[:, :12, 1] # only keep first 10 epochs
        #     losses = losses[:, -1]
        # else:
        losses = np.array([np.load(file_path) for file_path in file_paths_gt])
        # print(losses.shape)
        losses = losses[:, 5, 1]
        # print(losses.shape) 

        anytime_losses_diff.append(losses)

    anytime_losses_diff = np.array(anytime_losses_diff)
    print(anytime_losses_diff.shape)


    mean_losses_gt = np.mean(anytime_losses_gt, axis=0)
    mean_losses_diff = np.mean(anytime_losses_diff, axis=0)

    # print(mean_losses_gt[:10])
    # print(mean_losses_diff[:10])

    # losses_gt = anytime_losses_gt[:, 11, 1] # select test loss of last epoch for all cells in grid

    # anytime_losses_diff = np.array([np.load(file_path) for file_path in file_paths_diff[:]])
    # losses_diff = anytime_losses_diff[:, 11, 1] # select test loss of last epoch for all cells in grid
    # regret = np.where(mean_losses_diff < mean_losses_gt, 1, -1)
    better = 0
    for loss_gt, loss_diff in zip(mean_losses_gt, mean_losses_diff):
        if loss_diff < loss_gt:
            better += 1
    print(better)
    regret = (mean_losses_gt - mean_losses_diff) / mean_losses_gt
    # print('regret: ', regret)

    step_size= 0.1
    y_vals = np.arange(-1, 1 + step_size, step=step_size)
    z_vals = np.arange(-0.5, 0.5 + step_size, step=step_size)

    regret = np.reshape(regret, (len(z_vals), len(y_vals)))

    # # z-score
    # # regret = (regret - np.mean(regret)) / np.std(regret)
    
    #min/max
    regret = (regret - np.min(regret)) / (np.max(regret) - np.min(regret)) * 2 -1


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
    cax = axs.imshow(regret, cmap='coolwarm', extent=[y_vals[0],y_vals[-1],z_vals[-1],z_vals[0]])

    axs.plot(gt_coords[0][0]-0.005, gt_coords[0][1]-0.01, 'o', color='blue')
    axs.plot(gt_coords[1][0]-0.035, gt_coords[1][1]-0.035, 'o', color='blue')
    axs.plot(gt_coords[3][0]+0.05, gt_coords[3][1]+0.025, 'o', color='blue')
    axs.plot(gt_coords[4][0]-0.01, gt_coords[4][1]+0.04, 'o', color='blue')
    
    
    axs.set_xlabel(r'$\bar{y}^h$')
    axs.set_ylabel(r'$\bar{z}^h$')
    axs.invert_yaxis()
    fig.colorbar(cax, orientation='vertical', shrink=0.6)
    # # axs.plot(mean_timestamps_gt, mean_losses_gt, label='FAP')
    # # axs.fill_between(mean_timestamps_gt, mean_losses_gt+error_losses_gt, mean_losses_gt-error_losses_gt, alpha=0.15)
    
    # # axs.plot(mean_timestamps_diff, mean_losses_diff, label='diffusion')
    # # axs.fill_between(mean_timestamps_diff, mean_losses_diff+error_losses_diff, mean_losses_diff-error_losses_diff, alpha=0.15)
    
    # # axs.grid()
    # # axs.legend()
    # # axs.set_ylabel('Test loss [m]')
    # # axs.set_xlabel('time in s')

    fig.savefig(f'eval/eval_heatmap_v3_3e.pdf', dpi=200)