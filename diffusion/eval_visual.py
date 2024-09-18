import torch
import torch.nn as nn
import numpy as np

from diffusion.diffusion_model import DiffusionModel

from torchsummary import summary

if __name__ == '__main__':
    from pathlib import Path
    import pickle

    np.random.seed(2562)
    torch.manual_seed(2562)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_samples = 4

    model = DiffusionModel(device)
    model.load('conditioned_unet_80x80_1000_3256i_255.pth')

    # model = UNet(in_size=1, out_size=1, device=device)
    # model.to(device)

    # model.load_state_dict(torch.load('conditioned_unet_80x80_1000_3256i_255.pth', map_location=device))
    # model.eval()

    with open('data/FAP_combined.pickle', 'rb') as f:
        data = pickle.load(f)
    gt_patches = []
    gt_targets = []
    #gt_positions = []
    for i in range(n_samples):
        gt_patches.append(data[i][0])
        gt_targets.append(data[i][1])
        #gt_positions.append(data[i][2])

    gt_patches = np.array(gt_patches)[:n_samples]
    gt_targets = np.array(gt_targets)[:n_samples]


    # # running into memory issues with this sample function! fix: don't compute gradients
    with torch.no_grad():
        samples_1 = model.sample(n_samples, torch.tensor(gt_targets), device, patch_size=[80, 80], n_steps=1_000).detach().cpu().numpy()
        samples_2 = model.sample(n_samples, torch.tensor(gt_targets), device, patch_size=[80, 80], n_steps=1_000).detach().cpu().numpy()
        samples_3 = model.sample(n_samples, torch.tensor(gt_targets), device, patch_size=[80, 80], n_steps=1_000).detach().cpu().numpy()

    print(samples_1.shape)
    print(samples_2.shape)

    print(samples_1.max())
    print(samples_2.max())

    # samples = (samples - np.min(samples)) / (np.max(samples) - np.min(samples)) # normalize

    # print(gt_patches.max(), samples.max())

    # from skimage.metrics import structural_similarity as ssim

    # all_ssim = []
    # for gt, sample in zip(gt_patches, samples):
    #     all_ssim.append(ssim(gt, sample[0], data_range=sample.max() - sample.min()))

    # all_ssim = np.array(all_ssim)
    # np.save(f'eval/ssim_{n_samples}.npy', all_ssim)

    # print(all_ssim.shape)

    # print("SSIM mean: ", np.mean(all_ssim), ", std: ", np.std(all_ssim))
    # print("SSIM first 4: ", all_ssim[:4])

    
    # # print(np.min(samples), np.max(samples))

    import matplotlib.pyplot as plt
    plt.rcParams.update({
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.sans-serif": "Helvetica",
                "font.size": 12,
                "figure.figsize": (5, 5),
                "mathtext.fontset": 'stix'
    })

    # # random_idx = np.random.choice(len(gt_patches), size=2, replace=False)

    # # print(gt_targets[random_idx])
    fig, axs = plt.subplots(4, 4, layout='constrained')
    for j in range(4):
        axs[0, j].imshow(gt_patches[j], cmap='gray')
        axs[0, j].axis('off')
        # axs[0, j].set_title(f'{j + 1}')

    for j in range(4):
        axs[1, j].imshow(samples_1[j][0], cmap='gray')
        axs[1, j].axis('off')
        # axs[1, j].set_title(f'{j + 1}')

    for j in range(4):
        axs[2, j].imshow(samples_2[j][0], cmap='gray')
        axs[2, j].axis('off')
        # axs[2, j].set_title(f'Diffusion 2-{j + 1}')

    for j in range(4):
        axs[3, j].imshow(samples_3[j][0], cmap='gray')
        axs[3, j].axis('off')
        # axs[2, j].set_title(f'Diffusion 2-{j + 1}')


    # for i in range(2):
    #     for j in range(2):
    #         idx = i * 2 + j

    #         axs[i, j * 2].imshow(gt_patches[idx], cmap='gray')
    #         axs[i, j * 2].axis('off')
    #         axs[i, j * 2].set_title(f'FAP {idx + 1}')

    #         axs[i, j * 2 +1].imshow(samples[idx][0], cmap='gray')
    #         axs[i, j * 2 +1].axis('off')
    #         axs[i, j * 2 +1].set_title(f'diffusion {idx + 1}')

    # fig, axs = plt.subplots(2, 4, layout='constrained')
    # for i in range(4):
    #     axs[i][0]
    #     axs[i][1].imshow(samples[i], cmap='gray')
    # axs[0][0].imshow(gt_patches[random_idx[0]], cmap='gray')
    # axs[1][0].imshow(gt_patches[random_idx[1]], cmap='gray')
    # axs[0][1].imshow(diffusion_patches.detach().cpu().numpy()[random_idx[0]][0], cmap='gray')
    # axs[1][1].imshow(diffusion_patches.detach().cpu().numpy()[random_idx[1]][0], cmap='gray')

    # # plt.tight_layout()
    fig.savefig(f'eval/eval_visual_v2.pdf', dpi=200)