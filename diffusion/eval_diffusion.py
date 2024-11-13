import torch
import torch.nn as nn
import numpy as np

# from diffusion.diffusion_model import DiffusionModel
# from simulators.cf_frontnet_pt import CFSim

# from eval_gt_patches import calc_loss, get_targets, gen_T
import sys
sys.path.insert(0,'src/')
from attacks import calc_anytime_loss
# from create_dataset import read_data
from util import load_dataset
from create_dataset import load_targets, get_coeffs
from patch_placement import place_patch
from attacks import norm_transformation, get_transformation
import pickle

def calc_loss(dataset, patch, transformation_matrix, model, target, model_name='frontnet', quantized=False):
    with torch.no_grad():
        losses_per_image = []

        mask = torch.isnan(target)
        target = torch.where(mask, torch.tensor(0., dtype=torch.float32), target)

        for _, data in enumerate(dataset):
            img, _ = data
            img = img.to(patch.device) / 255. # limit images to range [0-1]
            
            mod_img = place_patch(img, patch, transformation_matrix)
            mod_img *= 255. # convert input images back to range [0-255.]
            mod_img.clamp_(0., 255.)
            if quantized:
                mod_img.floor_()

            if model_name == 'frontnet':
                # predict x, y, z, yaw
                x, y, z, phi = model(mod_img)

                # target_losses.append(torch.mean(all_l2))
                # prepare shapes for MSE loss
                # TODO: improve readbility!
                pred = torch.stack([x, y, z])
                pred = pred.squeeze(2).mT
            else:
                pred = model(mod_img)

            # only target x,y and z which are previously chosen, otherwise keep x/y/z to prediction
            #target_batch = torch.where(torch.isnan(target_batch), pred, target_batch)
            target = (pred * mask) + target
            
            loss = torch.nn.functional.mse_loss(target, pred)
            losses_per_image.append(loss.clone().detach().item())

        return np.array(losses_per_image)

def loss_dataset(model, gt_patch, diffusion_patch, random_patch, ft_patch, target, positions_gt, positions_ft, dataset, device, model_name='frontnet'):
    # T = gen_T(position)

    # gt_losses = []
    # diffusion_losses = []
    # random_losses = []
    # ft_losses = []

    # for data in dataset:
    #     img, _ = data

    # optimized_pos_vectors is of shape: [epochs, num_targets, num_patches, 3, 1]
    # print(positions_gt.shape)
    # positions_gt = torch.from_numpy(positions_gt).unsqueeze(1).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
    # positions_ft = torch.from_numpy(positions_ft).unsqueeze(2).unsqueeze(0).unsqueeze(0).to(device)
    gt_patch = torch.from_numpy(gt_patch).unsqueeze(0).unsqueeze(0).to(device)
    diffusion_patch = torch.from_numpy(diffusion_patch).unsqueeze(0).unsqueeze(0).to(device)
    random_patch = random_patch.unsqueeze(0).unsqueeze(0).to(device)
    ft_patch = torch.from_numpy(ft_patch).unsqueeze(0).unsqueeze(0).to(device)
    target = torch.from_numpy(target).unsqueeze(0).to(device)
    
    position_gt = torch.from_numpy(positions_gt).unsqueeze(1)
    matrix_gt = get_transformation(*position_gt).to(device)

    position_ft = torch.from_numpy(positions_ft.T)

    matrix_ft = get_transformation(*position_ft).to(device)



    # _, gt_loss = calc_anytime_loss(0, dataset, gt_patch, targets, model, positions_gt, scale_min=0.2, scale_max=0.7, quantized=False, model_name=model_name)
    # # gt_losses.append(calc_loss([targets], gt_patch, [T], img, sim))
    # _, diffusion_loss = calc_anytime_loss(0, dataset, diffusion_patch, targets, model, positions_gt, scale_min=0.2, scale_max=0.7, quantized=False, model_name=model_name)
    # # diffusion_losses.append(calc_loss([targets], diffusion_patch, [T], img, sim))
    # _, random_loss = calc_anytime_loss(0, dataset, random_patch, targets, model, positions_gt, scale_min=0.2, scale_max=0.7, quantized=False, model_name=model_name)
    # # random_losses.append(calc_loss([targets], random_patch, [T], img, sim))
    # _, ft_gt_loss = calc_anytime_loss(0, dataset, ft_patch, targets, model, positions_gt, scale_min=0.2, scale_max=0.7, quantized=False, model_name=model_name)
    # # ft_losses.append(calc_loss([targets], ft_patch, [T], img, sim))
    # _, ft_ft_loss = calc_anytime_loss(0, dataset, ft_patch, targets, model, positions_ft, scale_min=0.2, scale_max=0.7, quantized=False, model_name=model_name)


    gt_loss = calc_loss(dataset, gt_patch, matrix_gt, model, target, model_name=model_name)
    diffusion_loss = calc_loss(dataset, diffusion_patch, matrix_gt, model, target, model_name=model_name)
    random_loss = calc_loss(dataset, random_patch, matrix_gt, model, target, model_name=model_name)
    ft_gt_loss = calc_loss(dataset, ft_patch, matrix_gt, model, target, model_name=model_name)
    ft_ft_loss = calc_loss(dataset, ft_patch, matrix_ft, model, target, model_name=model_name)

    # print(gt_loss, diffusion_loss, random_loss, ft_gt_loss, ft_ft_loss)

    # return np.array(gt_losses), np.array(diffusion_losses), np.array(random_losses), np.array(ft_losses)
    return gt_loss, diffusion_loss, random_loss, ft_gt_loss, ft_ft_loss

def load_pickle(path):
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

def load_coeffs_from_settings(path):
    import yaml
    from attacks import norm_transformation
    parent_folder = path.parent
    with open(parent_folder / 'settings.yaml') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    coeffs = np.array(settings['patch']['position'], dtype=float).T
    coeffs = torch.from_numpy(coeffs)
    coeffs = torch.hstack(norm_transformation(*coeffs, scale_min=0.2, scale_max=0.7)).cpu().numpy()
    return coeffs

def read_from_folder(path, idx=-1):
    path = Path(path)

    file_paths = list(path.glob('[0-9]*/patches.npy'))
    file_paths.sort(key=lambda path: int(path.parent.name))

    # print(len(file_paths))

    # from util import load_dataset
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset_path = "pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"
    # dataset = load_dataset(dataset_path, batch_size=1, train=False, train_set_size=0.9)

    # if model == 'frontnet':
    #     from util import load_model
    #     model_path = 'pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    #     model_config = '160x32'
    #     model = load_model(path=model_path, device=device, config=model_config)
    #     model.eval()
    # else:
    #     from yolo_bounding import YOLOBox
    #     model = YOLOBox()
    #     model.to(device)

    # print(model)
    
    patches = np.array([np.load(file_path)[idx][0][0] for file_path in file_paths]) * 255.
    # print(patches.shape, patches.min(), patches.max())

    targets = np.array([load_targets(patch_path).numpy() for patch_path in file_paths])[:, 0, :]
    
    if idx == 0:
        coeffs = np.array([load_coeffs_from_settings(patch_path) for patch_path in file_paths])
    else:
        coeffs = np.array([get_coeffs(patch_path) for patch_path in file_paths])
    # print(coeffs.shape)
    # print(coeffs[:5])

    return patches, targets, coeffs


# def load_model(path, device):
#     model = UNet(in_size=1, out_size=1, device=device)
#     model.to(device)

#     model.load_state_dict(torch.load(path, map_location=device))
#     model.eval()

#     return model


if __name__ == '__main__':
    np.random.seed(2562)
    torch.manual_seed(2562)
    from tqdm import trange
    from pathlib import Path
    import pickle
    import os
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate patches')
    parser.add_argument('--model', type=str, choices=['frontnet', 'yolov5'], required=True, help='Model to use for evaluation')
    parser.add_argument('--calculate', action='store_true')
    parser.add_argument('--gt_dataset', type=str, default=None, help='Path to the ground truth patches pickle file')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to evaluate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_samples = 100


    if args.model == 'frontnet':
        from util import load_model
        model_path = 'pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
        model_config = '160x32'
        model = load_model(path=model_path, device=device, config=model_config)
        model.eval()
    elif args.model == 'yolov5':
        from yolo_bounding import YOLOBox
        model = YOLOBox()
    
    if args.calculate:
        dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
        test_set = load_dataset(path=dataset_path, batch_size=1, shuffle=False, drop_last=False, train=False, num_workers=0)

        print('Loading gt patches...')
        gt_patches, gt_targets, gt_positions = load_pickle(args.gt_dataset)

        # gt_patches, gt_targets, gt_positions = load_pickle('diffusion/yolopatches.pickle')
        # random_idx = np.random.choice(len(gt_patches), size=n_samples, replace=False)
        # np.save('results/yolo/fine-tuning80x80/indices.npy', random_idx)

        random_idx = np.load(f'results/finetuning/{args.model}/indices.npy')

        gt_patches = np.array(gt_patches[random_idx])
        # normalize
        gt_patches /= 255.
        
        gt_targets = np.array(gt_targets[random_idx])
        gt_positions = np.array(gt_positions[random_idx])

        # generate random patches
        print('Generating random patches...')
        random_patches = torch.rand(n_samples, 80, 80)
        print(random_patches.shape, random_patches.max())

        # load diffusion patches
        print('Loading diffusion patches...')
        diffusion_patches, diffusion_targets, diffusion_positions = read_from_folder(f'results/finetuning/{args.model}/hl_iter_10/', idx=0)
        diffusion_patches /= 255.
        print(diffusion_patches.shape, diffusion_patches.min(), diffusion_patches.max())

        # if not saved before, save as pickle
        # from src.create_dataset import save_pickle
        # save_pickle('results/finetuning/yolov5/hl_iter_10/', diffusion_patches, diffusion_targets, diffusion_positions)

        # load fine-tuned patches
        print('Loading fine-tuned patches...')
        ft_patches, ft_targets, ft_positions = read_from_folder(f'results/finetuning/{args.model}/hl_iter_10/', idx=-1)
        ft_patches /= 255.
        print(ft_patches.shape, ft_patches.min(), ft_patches.max())
        print(gt_patches.shape, gt_patches.min(), gt_patches.max())

        # # sanity check
        # print("Sanity check:")
        # print("Ground truth shapes:")
        # print(gt_patches.shape, gt_targets.shape, gt_positions.shape)
        # print(gt_targets[:5], gt_positions[:5]) 
        # print("Diffusion patches shapes:")
        # print(diffusion_patches.shape, diffusion_targets.shape, diffusion_positions.shape)
        # print("Fine-tuned patches shapes:")
        # print(ft_patches.shape, ft_targets.shape, ft_positions.shape)
        # print(ft_targets[:5], ft_positions[:5])
        

        all_gt = []
        all_diffusion = []
        all_random = []
        all_ft_at_gt = []
        all_ft_at_ft = []

        for i in trange(n_samples):
            gt_loss, diffusion_loss, random_loss, ft_at_gt_loss, ft_at_ft_loss = loss_dataset(model, gt_patches[i], diffusion_patches[i], random_patches[i], ft_patches[i], gt_targets[i], gt_positions[i], ft_positions[i], test_set, device, model_name=args.model)
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

        print("Final shapes: ", all_gt.shape, all_diffusion.shape, all_random.shape, all_ft_at_gt.shape, all_ft_at_ft.shape)


        os.makedirs(f'eval/{args.model}', exist_ok=True)
        np.save(f'eval/{args.model}/comparison_gt_{n_samples}.npy', all_gt)
        np.save(f'eval/{args.model}/comparison_diffusion_{n_samples}.npy', all_diffusion)
        np.save(f'eval/{args.model}/comparison_random_{n_samples}.npy', all_random)
        np.save(f'eval/{args.model}/comparison_ft@gt_{n_samples}.npy', all_ft_at_gt)
        np.save(f'eval/{args.model}/comparison_ft@ft_{n_samples}.npy', all_ft_at_ft)

    else:
        all_gt = np.load(f'eval/{args.model}/comparison_gt_100.npy')
        all_diffusion = np.load(f'eval/{args.model}/comparison_diffusion_100.npy')
        all_random = np.load(f'eval/{args.model}/comparison_random_100.npy')
        all_ft_at_gt = np.load(f'eval/{args.model}/comparison_ft@gt_100.npy')
        all_ft_at_ft = np.load(f'eval/{args.model}/comparison_ft@ft_100.npy')


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

    print(all_random.shape, all_gt.shape, all_diffusion.shape, all_ft_at_gt.shape, all_ft_at_ft.shape)

    data = np.array([np.mean(all_random.T, axis=1), np.mean(all_gt.T, axis=1), np.mean(all_diffusion.T, axis=1), np.mean(all_ft_at_gt.T, axis=1), np.mean(all_ft_at_ft.T, axis=1)])
    # data = np.array([all_random, all_gt, all_diffusion, all_ft_at_gt, all_ft_at_ft])
    # data = np.log10(data)
    print(data.shape)

    import matplotlib.pyplot as plt

    # change settings to match latex
    # try:
    plt.rcParams.update({
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.sans-serif": "Helvetica",
                "font.size": 12,
                "figure.figsize": (5, 3),
                "mathtext.fontset": 'stix'
    })
    # except FileNotFoundError:
    #     import shlex
    #     import subprocess
    #     command = shlex.split('module load texlive')
    #     subprocess.run(command)

    fig, axs = plt.subplots(1, 1, layout='constrained')
    axs.boxplot(data.T, 'D', tick_labels=['random', 'ground truth', 'diffusion', 'fine tuning*', 'fine tuning+']) 
    axs.set_yscale('log')
    axs.set_ylabel('Test loss [m]')
    plt.grid(True, which="both", ls="-", color='0.65')

    fig.savefig(f'eval/{args.model}/eval_boxplot_{n_samples}.pdf', dpi=200)
    
    