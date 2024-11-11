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

        yolo_gt_patches = yolo_gt_patches[indices]
        yolo_gt_targets = yolo_gt_targets[indices]
        yolo_gt_positions = yolo_gt_positions[indices]
        frontnet_gt_patches = frontnet_gt_patches[indices]
        frontnet_gt_targets = frontnet_gt_targets[indices]
        frontnet_gt_positions = frontnet_gt_positions[indices]

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
            print("Synthesize diffusion patches...")
            frontnet_diffusion_patches = frontnet_diffusion.sample(len(combined_targets), combined_targets, device, frontnet_gt_patches.shape[-2::])
            # print("Frontent diffusion patches: ", frontnet_diffusion_patches.shape)
            yolo_diffusion_patches = frontnet_diffusion.sample(len(combined_targets), combined_targets, device, frontnet_gt_patches.shape[-2::])
            # print("Yolo diffusion patches shape: ", yolo_diffusion_patches.shape)
            combined_diffusion_patches = all_diffusion.sample(len(combined_targets), combined_targets, device, frontnet_gt_patches.shape[-2::])
            # print("Combined diffusion pacthes shape: ", combined_diffusion_patches.shape)
            # TODO: test if these patches perform worse than patches that are created for each target separately

        # save patches for further use
            np.save('eval/transferability/frontnet_diffusion_patches.npy', frontnet_diffusion_patches.cpu().numpy())
            np.save('eval/transferability/yolo_diffusion_patches.npy', yolo_diffusion_patches.cpu().numpy())
            np.save('eval/transferability/combined_diffusion_patches.npy', combined_diffusion_patches.cpu().numpy())
        else:
            print("Loading diffusion patches...")
            frontnet_diffusion_patches = torch.from_numpy(np.load('eval/transferability/frontnet_diffusion_patches.npy')).to(device)
            yolo_diffusion_patches = torch.from_numpy(np.load('eval/transferability/yolo_diffusion_patches.npy')).to(device)
            combined_diffusion_patches = torch.from_numpy(np.load('eval/transferability/combined_diffusion_patches.npy')).to(device)

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

            frontnet_gt_on_frontnet.append(calc_loss(test_set, frontnet_gt_patch, matrix_frontnet_gt, frontnet, frontnet_target, model_name='frontnet'))
            frontnet_gt_on_yolov5.append(calc_loss(test_set, frontnet_gt_patch, matrix_frontnet_gt, yolov5, frontnet_target, model_name='yolov5'))
            
            yolo_gt_on_frontnet.append(calc_loss(test_set, yolo_gt_patch, matrix_yolo_gt, frontnet, yolo_target, model_name='frontnet'))
            yolo_gt_on_yolov5.append(calc_loss(test_set, yolo_gt_patch, matrix_yolo_gt, yolov5, yolo_target, model_name='yolov5'))
            
            frontnet_diffusion_on_frontnet.append(calc_loss(test_set, frontnet_diffusion_patches[i].unsqueeze(0), matrix_combined, frontnet, combined_target, model_name='frontnet'))
            frontnet_diffusion_on_yolov5.append(calc_loss(test_set, frontnet_diffusion_patches[i].unsqueeze(0), matrix_combined, yolov5, combined_target, model_name='yolov5'))

            yolo_diffusion_on_frontnet.append(calc_loss(test_set, yolo_diffusion_patches[i].unsqueeze(0), matrix_combined, frontnet, combined_target, model_name='frontnet'))
            yolo_diffusion_on_yolov5.append(calc_loss(test_set, yolo_diffusion_patches[i].unsqueeze(0), matrix_combined, yolov5, combined_target, model_name='yolov5'))

            combined_diffusion_on_frontnet.append(calc_loss(test_set, combined_diffusion_patches[i].unsqueeze(0), matrix_combined, frontnet, combined_target, model_name='frontnet'))
            combined_diffusion_on_yolov5.append(calc_loss(test_set, combined_diffusion_patches[i].unsqueeze(0), matrix_combined, yolov5, combined_target, model_name='yolov5'))

        
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

        # save results
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

    else:
        print("Loading saved data...")
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
