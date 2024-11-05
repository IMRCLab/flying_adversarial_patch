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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate transferability of patches')
    # parser.add_argument('--model', type=str, choices=['frontnet', 'yolov5'], required=True, help='Model to use for evaluation')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading frontnet...")
    model_path = 'pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'
    frontnet = load_model(path=model_path, device=device, config=model_config)
    frontnet.eval()

    print("Loading yolo...")
    yolov5 = YOLOBox()
    
    print("Loading diffusion models...")
    frontnet_diffusion = DiffusionModel(device)
    frontnet_diffusion.load(f'diffusion/frontnet_diffusion.pth')

    yolov5_diffusion = DiffusionModel(device)
    yolov5_diffusion.load(f'diffusion/yolov5_diffusion.pth')

    all_diffusion = DiffusionModel(device)
    all_diffusion.load(f'diffusion/all_diffusion.pth')


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
    np.testing.assert_array_equal(yolo_gt_targets, frontnet_gt_targets)
    

    frontnet_gt_on_frontnet = []
    frontnet_gt_on_yolov5 = []
    yolo_gt_on_frontnet = []
    yolo_gt_on_yolov5 = []
    all_on_frontnet = []
    all_on_yolo = []

    for i in range(1):#len(indices)):
        target = torch.from_numpy(frontnet_gt_targets[i]).unsqueeze(0).to(device)
        print(target)
        frontnet_position = torch.from_numpy(frontnet_gt_positions[i]).unsqueeze(1)
        print(frontnet_position)
        matrix_gt = get_transformation(*frontnet_position).to(device)
        print(matrix_gt)
        
        frontnet_gt_on_frontnet.append(calc_loss(test_set, frontnet_gt_patches[i], matrix_gt, frontnet, ))

    # gt_patch = torch.from_numpy(gt_patch).unsqueeze(0).unsqueeze(0).to(device)
    # diffusion_patch = torch.from_numpy(diffusion_patch).unsqueeze(0).unsqueeze(0).to(device)
    # random_patch = random_patch.unsqueeze(0).unsqueeze(0).to(device)
    # ft_patch = torch.from_numpy(ft_patch).unsqueeze(0).unsqueeze(0).to(device)
    # target = torch.from_numpy(target).unsqueeze(0).to(device)
    
    # position_gt = torch.from_numpy(positions_gt).unsqueeze(1)
    # matrix_gt = get_transformation(*position_gt).to(device)


