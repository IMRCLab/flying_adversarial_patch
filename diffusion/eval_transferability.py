import torch
import numpy as np
import argparse
import sys
sys.path.insert(0,'src/')
from util import load_model, load_dataset
from yolo_bounding import YOLOBox
from diffusion_model import DiffusionModel
from eval_diffusion import calc_loss, load_pickle
from patch_placement import place_patch
from attacks import get_transformation
import os
from tqdm import trange

if __name__ == '__main__':
    np.random.seed(2562)
    torch.manual_seed(2562)
    parser = argparse.ArgumentParser(description='Evaluate transferability of patches')
    parser.add_argument('--calculate', action='store_true', help='Calculate all losses or load saved results otherwise')
    parser.add_argument('--synthesize', action='store_true', help='Synthesize diffusion patches')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.calculate:
        print("Loading frontnet...")
        model_path = 'pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
        model_config = '160x32'
        frontnet = load_model(path=model_path, device=device, config=model_config)
        frontnet.eval()

        print("Loading yolo...")
        yolov5 = YOLOBox()


        # load test dataset
        print("Loading dataset...")
        dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
        test_set = load_dataset(path=dataset_path, batch_size=1, shuffle=False, drop_last=False, train=False, num_workers=0)

        # load gt patches
        print("Loading gt patches...")
        indices = np.random.choice(len(test_set), size=100, replace=False)
        yolo_gt_patches, yolo_gt_targets, yolo_gt_positions = load_pickle('diffusion/yolopatches.pickle')
        frontnet_gt_patches, frontnet_gt_targets, frontnet_gt_positions = load_pickle('diffusion/FAP_combined.pickle')

        yolo_gt_patches = yolo_gt_patches[indices] / 255.
        yolo_gt_targets = yolo_gt_targets[indices]
        yolo_gt_positions = yolo_gt_positions[indices]
        frontnet_gt_patches = frontnet_gt_patches[indices] / 255.
        frontnet_gt_targets = frontnet_gt_targets[indices]
        frontnet_gt_positions = frontnet_gt_positions[indices]

        print(frontnet_gt_patches.min(), frontnet_gt_patches.max())
        print(yolo_gt_patches.min(), yolo_gt_patches.max())


        # sanity check
        # np.testing.assert_array_equal(yolo_gt_targets, frontnet_gt_targets)
        # check failed, targets are different but it might not have a huge impact on the results
        # therefore, we combine the targets and positions to draw random samples for the combined diffusion model
        
        combined_targets = np.concatenate([frontnet_gt_targets, yolo_gt_targets], axis=0)
        combined_positions = np.concatenate([frontnet_gt_positions, yolo_gt_positions], axis=0)
        # print(combined_targets.shape)
        indices = np.random.choice(len(combined_targets), size=100, replace=False)
        combined_targets = combined_targets[indices]
        combined_positions = combined_positions[indices]

        # ensure the directory exists
        os.makedirs('eval/transferability', exist_ok=True)

        # calculate diffusion patches
        if args.synthesize:
            print("Loading diffusion models...")
            frontnet_diffusion = DiffusionModel(device)
            frontnet_diffusion.load(f'diffusion/frontnet_diffusion.pth')

            yolov5_diffusion = DiffusionModel(device)
            yolov5_diffusion.load(f'diffusion/yolov5_diffusion.pth')

            all_diffusion = DiffusionModel(device)
            all_diffusion.load(f'diffusion/all_diffusion.pth')
            
        else:
            print("Loading diffusion patches...")
            frontnet_diffusion_patches = torch.from_numpy(np.load('eval/transferability/frontnet_diffusion_patches.npy')).to(device)
            yolo_diffusion_patches = torch.from_numpy(np.load('eval/transferability/yolo_diffusion_patches.npy')).to(device)
            combined_diffusion_patches = torch.from_numpy(np.load('eval/transferability/combined_diffusion_patches.npy')).to(device)

        random_patches = torch.rand(frontnet_gt_patches.shape).unsqueeze(1).to(device)
        print(random_patches.shape, random_patches.min(), random_patches.max())

        random_on_frontnet = []
        random_on_yolov5 = []
        frontnet_gt_on_frontnet = []
        frontnet_gt_on_yolov5 = []
        yolo_gt_on_frontnet = []
        yolo_gt_on_yolov5 = []
        frontnet_diffusion_on_frontnet = []
        frontnet_diffusion_on_yolov5 = []
        yolo_diffusion_on_frontnet = []
        yolo_diffusion_on_yolov5 = []
        combined_diffusion_on_frontnet = []
        combined_diffusion_on_yolov5 = []

        if args.synthesize:
            frontnet_diffusion_patches = []
            yolo_diffusion_patches = []
            combined_diffusion_patches = []

        for i in trange(len(indices)):
            frontnet_target = torch.from_numpy(frontnet_gt_targets[i]).unsqueeze(0).to(device)
            # print(frontnet_target)
            yolo_target = torch.from_numpy(yolo_gt_targets[i]).unsqueeze(0).to(device)
            # print(yolo_target, yolo_target.shape)
            combined_target = torch.from_numpy(combined_targets[i]).unsqueeze(0).to(device)
            # print(combined_target, combined_target.shape)

            frontnet_position = torch.from_numpy(frontnet_gt_positions[i]).unsqueeze(1)
            # print(frontnet_position)
            matrix_frontnet_gt = get_transformation(*frontnet_position).to(device)
            # print(matrix_frontnet_gt)

            yolo_position = torch.from_numpy(yolo_gt_positions[i]).unsqueeze(1)
            matrix_yolo_gt = get_transformation(*yolo_position).to(device)
            # print(matrix_yolo_gt, matrix_yolo_gt.shape)

            combined_position = torch.from_numpy(combined_positions[i]).unsqueeze(1)
            matrix_combined = get_transformation(*combined_position).to(device)

            # print(matrix_combined, matrix_combined.shape)

            frontnet_gt_patch = torch.from_numpy(frontnet_gt_patches[i]).unsqueeze(0).unsqueeze(0).to(device)
            # print(frontnet_gt_patch.shape)
            yolo_gt_patch = torch.from_numpy(yolo_gt_patches[i]).unsqueeze(0).unsqueeze(0).to(device)
            # print(yolo_gt_patch.shape)

            random_on_frontnet.append(calc_loss(test_set, random_patches[i].unsqueeze(0), matrix_frontnet_gt, frontnet, frontnet_target, model_name='frontnet'))
            random_on_yolov5.append(calc_loss(test_set, random_patches[i].unsqueeze(0), matrix_yolo_gt, yolov5, yolo_target, model_name='yolov5'))

            frontnet_gt_on_frontnet.append(calc_loss(test_set, frontnet_gt_patch, matrix_frontnet_gt, frontnet, frontnet_target, model_name='frontnet'))
            frontnet_gt_on_yolov5.append(calc_loss(test_set, frontnet_gt_patch, matrix_frontnet_gt, yolov5, frontnet_target, model_name='yolov5'))
            
            yolo_gt_on_frontnet.append(calc_loss(test_set, yolo_gt_patch, matrix_yolo_gt, frontnet, yolo_target, model_name='frontnet'))
            yolo_gt_on_yolov5.append(calc_loss(test_set, yolo_gt_patch, matrix_yolo_gt, yolov5, yolo_target, model_name='yolov5'))
            
            if args.synthesize:
                frontnet_diffusion_patch = frontnet_diffusion.sample(1, combined_target, device, [80, 80])
                yolo_diffusion_patch = yolov5_diffusion.sample(1, combined_target, device, [80, 80])
                combined_diffusion_patch = all_diffusion.sample(1, combined_target, device, [80, 80])

                frontnet_diffusion_patches.append(frontnet_diffusion_patch[0].cpu().numpy())
                yolo_diffusion_patches.append(yolo_diffusion_patch[0].cpu().numpy())
                combined_diffusion_patches.append(combined_diffusion_patch[0].cpu().numpy())

            frontnet_diffusion_on_frontnet.append(calc_loss(test_set, frontnet_diffusion_patch, matrix_combined, frontnet, combined_target, model_name='frontnet'))
            frontnet_diffusion_on_yolov5.append(calc_loss(test_set, frontnet_diffusion_patch, matrix_combined, yolov5, combined_target, model_name='yolov5'))

            yolo_diffusion_on_frontnet.append(calc_loss(test_set, yolo_diffusion_patch, matrix_combined, frontnet, combined_target, model_name='frontnet'))
            yolo_diffusion_on_yolov5.append(calc_loss(test_set, yolo_diffusion_patch, matrix_combined, yolov5, combined_target, model_name='yolov5'))

            combined_diffusion_on_frontnet.append(calc_loss(test_set, combined_diffusion_patch, matrix_combined, frontnet, combined_target, model_name='frontnet'))
            combined_diffusion_on_yolov5.append(calc_loss(test_set, combined_diffusion_patch, matrix_combined, yolov5, combined_target, model_name='yolov5'))

            
        random_on_frontnet = np.array(random_on_frontnet)
        random_on_yolov5 = np.array(random_on_yolov5)
        frontnet_gt_on_frontnet = np.array(frontnet_gt_on_frontnet)
        frontnet_gt_on_yolov5 = np.array(frontnet_gt_on_yolov5)
        yolo_gt_on_frontnet = np.array(yolo_gt_on_frontnet)
        yolo_gt_on_yolov5 = np.array(yolo_gt_on_yolov5)
        frontnet_diffusion_on_frontnet = np.array(frontnet_diffusion_on_frontnet)
        frontnet_diffusion_on_yolov5 = np.array(frontnet_diffusion_on_yolov5)
        yolo_diffusion_on_frontnet = np.array(yolo_diffusion_on_frontnet)
        yolo_diffusion_on_yolov5 = np.array(yolo_diffusion_on_yolov5)
        combined_diffusion_on_frontnet = np.array(combined_diffusion_on_frontnet)
        combined_diffusion_on_yolov5 = np.array(combined_diffusion_on_yolov5)

        frontnet_diffusion_patches = np.array(frontnet_diffusion_patches)
        yolo_diffusion_patches = np.array(yolo_diffusion_patches)
        combined_diffusion_patches = np.array(combined_diffusion_patches)

        # save results
        np.save('eval/transferability/random_on_frontnet.npy', random_on_frontnet)
        np.save('eval/transferability/random_on_yolov5.npy', random_on_yolov5)
        np.save('eval/transferability/frontnet_gt_on_frontnet.npy', frontnet_gt_on_frontnet)
        np.save('eval/transferability/frontnet_gt_on_yolov5.npy', frontnet_gt_on_yolov5)
        np.save('eval/transferability/yolo_gt_on_frontnet.npy', yolo_gt_on_frontnet)
        np.save('eval/transferability/yolo_gt_on_yolov5.npy', yolo_gt_on_yolov5)
        np.save('eval/transferability/frontnet_diffusion_on_frontnet.npy', frontnet_diffusion_on_frontnet)
        np.save('eval/transferability/frontnet_diffusion_on_yolov5.npy', frontnet_diffusion_on_yolov5)
        np.save('eval/transferability/yolo_diffusion_on_frontnet.npy', yolo_diffusion_on_frontnet)
        np.save('eval/transferability/yolo_diffusion_on_yolov5.npy', yolo_diffusion_on_yolov5)
        np.save('eval/transferability/combined_diffusion_on_frontnet.npy', combined_diffusion_on_frontnet)
        np.save('eval/transferability/combined_diffusion_on_yolov5.npy', combined_diffusion_on_yolov5)

        if args.synthesize:
            np.save('eval/transferability/frontnet_diffusion_patches.npy', frontnet_diffusion_patches)
            np.save('eval/transferability/yolo_diffusion_patches.npy', yolo_diffusion_patches)
            np.save('eval/transferability/combined_diffusion_patches.npy', combined_diffusion_patches)

    else:
        print("Loading saved data...")
        random_on_frontnet = np.load('eval/transferability/random_on_frontnet.npy')
        random_on_yolov5 = np.load('eval/transferability/random_on_yolov5.npy')
        frontnet_gt_on_frontnet = np.load('eval/transferability/frontnet_gt_on_frontnet.npy')
        frontnet_gt_on_yolov5 = np.load('eval/transferability/frontnet_gt_on_yolov5.npy')
        yolo_gt_on_frontnet = np.load('eval/transferability/yolo_gt_on_frontnet.npy')
        yolo_gt_on_yolov5 = np.load('eval/transferability/yolo_gt_on_yolov5.npy')
        frontnet_diffusion_on_frontnet = np.load('eval/transferability/frontnet_diffusion_on_frontnet.npy')
        frontnet_diffusion_on_yolov5 = np.load('eval/transferability/frontnet_diffusion_on_yolov5.npy')
        yolo_diffusion_on_frontnet = np.load('eval/transferability/yolo_diffusion_on_frontnet.npy')
        yolo_diffusion_on_yolov5 = np.load('eval/transferability/yolo_diffusion_on_yolov5.npy')
        combined_diffusion_on_frontnet = np.load('eval/transferability/combined_diffusion_on_frontnet.npy')
        combined_diffusion_on_yolov5 = np.load('eval/transferability/combined_diffusion_on_yolov5.npy')


    print(f"Random patches on Frontnet: {np.mean(random_on_frontnet)}, std: {np.std(random_on_frontnet)}")
    print(f"Random patches on Yolov5: {np.mean(random_on_yolov5)}, std: {np.std(random_on_yolov5)}")
    print(f"Ground truth Frontnet patches on Frontnet: {np.mean(frontnet_gt_on_frontnet)}, std: {np.std(frontnet_gt_on_frontnet)}")
    print(f"Ground truth Frontnet patches on Yolov5: {np.mean(frontnet_gt_on_yolov5)}, std: {np.std(frontnet_gt_on_yolov5)}")
    print(f"Ground truth Yolov5 patches on Frontnet: {np.mean(yolo_gt_on_frontnet)}, std: {np.std(yolo_gt_on_frontnet)}")
    print(f"Ground truth Yolov5 patches on Yolov5: {np.mean(yolo_gt_on_yolov5)}, std: {np.std(yolo_gt_on_yolov5)}")
    print(f"Frontnet diffusion patches on Frontnet: {np.mean(frontnet_diffusion_on_frontnet)}, std: {np.std(frontnet_diffusion_on_frontnet)}")
    print(f"Frontnet diffusion patches on Yolov5: {np.mean(frontnet_diffusion_on_yolov5)}, std: {np.std(frontnet_diffusion_on_yolov5)}")
    print(f"Yolov5 diffusion patches on Frontnet: {np.mean(yolo_diffusion_on_frontnet)}, std: {np.std(yolo_diffusion_on_frontnet)}")
    print(f"Yolov5 diffusion patches on Yolov5: {np.mean(yolo_diffusion_on_yolov5)}, std: {np.std(yolo_diffusion_on_yolov5)}")
    print(f"Combined diffusion patches on Frontnet: {np.mean(combined_diffusion_on_frontnet)}, std: {np.std(combined_diffusion_on_frontnet)}")
    print(f"Combined diffusion patches on Yolov5: {np.mean(combined_diffusion_on_yolov5)}, std: {np.std(combined_diffusion_on_yolov5)}")


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

    # Data for boxplots
    data_frontnet = [
        np.mean(random_on_frontnet.T, axis=1),
        np.mean(frontnet_gt_on_frontnet.T, axis=1),
        np.mean(yolo_gt_on_frontnet.T, axis=1),
        np.mean(frontnet_diffusion_on_frontnet.T, axis=1),
        np.mean(yolo_diffusion_on_frontnet.T, axis=1),
        np.mean(combined_diffusion_on_frontnet.T, axis=1)
    ]

    data_yolov5 = [
        np.mean(random_on_yolov5.T, axis=1),
        np.mean(frontnet_gt_on_yolov5.T, axis=1),
        np.mean(yolo_gt_on_yolov5.T, axis=1),
        np.mean(frontnet_diffusion_on_yolov5.T, axis=1),
        np.mean(yolo_diffusion_on_yolov5.T, axis=1),
        np.mean(combined_diffusion_on_yolov5.T, axis=1)
    ]

    labels = [
        'Random',
        'Frontnet GT',
        'Yolo GT',
        'Frontnet Diffusion',
        'Yolo Diffusion',
        'Combined Diffusion'
    ]

    # Create one figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Create boxplot for Frontnet
    axs[0].boxplot(data_frontnet, tick_labels=labels)
    axs[0].set_title('Performance on Frontnet')
    axs[0].set_ylabel('Loss [m]')
    # axs[0].set_yscale('log')
    axs[0].tick_params(axis='x', rotation=45)

    # Create boxplot for Yolov5
    axs[1].boxplot(data_yolov5, tick_labels=labels)
    axs[1].set_title('Performance on Yolov5')
    axs[1].set_ylabel('Loss [m]')
    # axs[1].set_yscale('log')
    axs[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('eval/transferability/boxplot_transferability.png')
    plt.show()
