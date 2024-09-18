import torch
import torch.nn as nn
import numpy as np

from diffusion.diffusion_model import DiffusionModel
# from simulators.cf_frontnet_pt import CFSim

# from eval_gt_patches import calc_loss, get_targets, gen_T
from src.attacks import calc_anytime_loss

def loss_dataset(model, gt_patch, diffusion_patch, random_patch, ft_patch, targets, positions_gt, positions_ft, dataset):
    # T = gen_T(position)

    # gt_losses = []
    # diffusion_losses = []
    # random_losses = []
    # ft_losses = []

    # for data in dataset:
    #     img, _ = data
        _, gt_loss = calc_anytime_loss(0, dataset, gt_patch, targets, model, positions_gt, scale_min=0.2, scale_max=0.7, quantized=False)
        # gt_losses.append(calc_loss([targets], gt_patch, [T], img, sim))
        _, diffusion_loss = calc_anytime_loss(0, dataset, diffusion_patch, targets, model, positions_gt, scale_min=0.2, scale_max=0.7, quantized=False)
        # diffusion_losses.append(calc_loss([targets], diffusion_patch, [T], img, sim))
        _, random_loss = calc_anytime_loss(0, dataset, random_patch, targets, model, positions_gt, scale_min=0.2, scale_max=0.7, quantized=False)
        # random_losses.append(calc_loss([targets], random_patch, [T], img, sim))
        _, ft_gt_loss = calc_anytime_loss(0, dataset, ft_patch, targets, model, positions_gt, scale_min=0.2, scale_max=0.7, quantized=False)
        # ft_losses.append(calc_loss([targets], ft_patch, [T], img, sim))
        _, ft_ft_loss = calc_anytime_loss(0, dataset, ft_patch, targets, model, positions_ft, scale_min=0.2, scale_max=0.7, quantized=False)

    # return np.array(gt_losses), np.array(diffusion_losses), np.array(random_losses), np.array(ft_losses)
    return gt_loss.detach().cpu().numpy(), diffusion_loss.detach().cpu().numpy(), random_loss.detach().cpu().numpy(), ft_gt_loss.detach().cpu().numpy(), ft_ft_loss.detach().cpu().numpy()

def load_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    patches = []
    targets = []
    positions = []
    for i in range(len(data)):
        patches.append(data[i][0])
        targets.append(data[i][1])
        positions.append(data[i][2])

    return np.array(patches)*255., np.array(targets), np.array(positions)

# def load_model(path, device):
#     model = UNet(in_size=1, out_size=1, device=device)
#     model.to(device)

#     model.load_state_dict(torch.load(path, map_location=device))
#     model.eval()

#     return model


if __name__ == '__main__':
    from pathlib import Path
    import pickle

    np.random.seed(2562)
    torch.manual_seed(2562)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_samples = 100

    model = DiffusionModel(device)
    model.load('/home/hanfeld/bb_FAP/conditioned_unet_80x80_1000_3256i_255.pth')

    
    random_idx = np.random.choice(len(gt_patches), size=n_samples, replace=False)
    np.save('results/fine-tuning80x80/indices.npy', random_idx)

    gt_patches, gt_targets, gt_positions = load_dataset('diffusion/FAP_combined.pickle')
    gt_patches = gt_patches[random_idx]
    gt_targets = gt_targets[random_idx]
    gt_positions = gt_positions[random_idx]



    with torch.no_grad():
        diffusion_patches = model.sample(n_samples, torch.tensor(gt_targets), device, patch_size=[80,80], n_steps=1_000).detach().cpu()

    # # # # print(diffusion_patches.shape)
    # # # # # print(torch.min(diffusion_patches), torch.max(diffusion_patches))
    # diffusion_patches = (diffusion_patches - torch.min(diffusion_patches)) / (torch.max(diffusion_patches) - torch.min(diffusion_patches)) # normalize
    diffusion_patches *= 255.

    # print(diffusion_patches.shape)
    diffusion_patches = diffusion_patches.detach().cpu().numpy()
    # np.save('eval/diffusion_patches_100.npy', diffusion_patches)
    # # # print(torch.min(diffusion_patches), torch.max(diffusion_patches))
    # # # print(diffusion_patches.shape)

    # diffusion_patches = torch.tensor(np.load('eval/diffusion_patches_100.npy'))
    # # print(diffusion_patches.shape)

    np.save(f'eval/diffusion_patches_{n_samples}.npy', diffusion_patches.numpy())

    # # # import matplotlib.pyplot as plt
    # # # fig = plt.figure(constrained_layout=True)
    # # # n_cols = min(n_samples, 5)
    # # # n_rows = int(np.ceil(n_samples / 5))
    # # # axs_samples = fig.subplots(n_rows, n_cols).flatten()
    # # # for i, sample in enumerate(diffusion_patches):
    # # #     axs_samples[i].imshow(sample[0], cmap='gray')
    # # #     axs_samples[i].axis('off')
    # # #     # axs_samples[i].set_title(f'sample {i}')
    # # # fig.savefig(f'eval/diffusion_patches_{n_samples}.png', dpi=200)
    # # # plt.show()

    # sim = CFSim()
    
    # generate random patches
    random_patches = torch.rand_like(diffusion_patches) * 255.
    random_patches = torch.rand(n_samples, 80, 80) * 255.
    print(random_patches.shape, random_patches.max())

    ft_patches, ft_targets, ft_positions = load_dataset('diffusion/FAP_fine_tuning.pickle')


    all_gt, all_diffusion, all_random, all_ft_at_gt, all_ft_at_ft = []

    from tqdm import trange

    for i in trange(n_samples):
        gt_loss, diffusion_loss, random_loss, ft_at_gt_loss, ft_at_ft_loss = loss_dataset(model, gt_patches, diffusion_patch, random_patch, ft_patch, targets, positions_gt, positions_ft, dataset, sim)
        all_gt.append(gt_loss)
        all_diffusion.append(diffusion_loss)
        all_random.append(random_loss)
        all_ft_at_gt.append(ft_at_gt_loss)
        all_ft_at_ft.append(ft_at_ft_loss)

    all_gt = np.array(all_gt)
    all_diffusion = np.array(all_diffusion)
    all_random = np.array(all_random)
    all_ft_at_gt = np.array(all_ft_at_gt)
    all_ft_at_ft = np.array(all_ft_at_ft)

    # np.save(f'eval/comparison_gt_{n_samples}.npy', all_gt)
    # np.save(f'eval/comparison_diffusion_{n_samples}.npy', all_diffusion)
    # np.save(f'eval/comparison_random_{n_samples}.npy', all_random)
    # np.save(f'eval/comparison_ft@gt_{n_samples}.npy', all_ft_at_gt)
    # np.save(f'eval/comparison_ft@ft_{n_samples}.npy', all_ft_at_ft)

    all_gt = np.load('eval/comparison_gt_100.npy')
    all_diffusion = np.load('eval/comparison_diffusion_100.npy')
    all_random = np.load('eval/comparison_random_100.npy')
    all_ft_at_gt = np.load('eval/comparison_ft@gt_100.npy')
    all_ft_at_ft = np.load('eval/comparison_ft@ft_100.npy')


    print(f"Ground truth patches mean loss: {np.mean(all_gt)}, std: {np.std(all_gt)}")
    # print(f"Per patch, mean: {np.mean(all_gt, axis=1)}, std: {np.std(all_gt, axis=1)}")
    print()
    print(f"Random patches mean loss: {np.mean(all_random)}, std: {np.std(all_random)}")
    # print(f"Per patch, mean: {np.mean(all_random, axis=1)}, std: {np.std(all_random, axis=1)}")
    print()
    print(f"Diffusion patches mean loss: {np.mean(all_diffusion)}, std: {np.std(all_diffusion)}")
    # print(f"Per patch, mean: {np.mean(all_diffusion, axis=1)}, std: {np.std(all_diffusion, axis=1)}")
    print()
    print(f"Fine-tuning @ gt pos patches mean loss: {np.mean(all_ft_at_gt)}, std: {np.std(all_ft_at_gt)}")
    print()
    print(f"Fine-tuning @ ft pos patches mean loss: {np.mean(all_ft_at_ft)}, std: {np.std(all_ft_at_ft)}")
    print()


    data = np.array([np.mean(all_random.T, axis=1), np.mean(all_gt.T, axis=1), np.mean(all_diffusion.T, axis=1), np.mean(all_ft_at_gt.T, axis=1), np.mean(all_ft_at_ft.T, axis=1)])
    # data = np.log10(data)
    print(data.shape)

    import matplotlib.pyplot as plt

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
    axs.boxplot(data.T, 'D', tick_labels=['random', 'ground truth', 'diffusion', 'fine tuning*', 'fine tuning+']) 
    axs.set_yscale('log')
    axs.set_ylabel('Test loss [m]')
    plt.grid(True, which="both", ls="-", color='0.65')

    fig.savefig(f'eval/eval_boxplot_{n_samples}_2.pdf', dpi=200)
    
    