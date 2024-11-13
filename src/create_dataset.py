import yaml
import subprocess 
import numpy as np
from pathlib import Path
import os
import shlex
import pickle
import argparse

import torch
from patch_placement import place_patch
from util import load_model, load_dataset

from yolo_bounding import YOLOBox


def gen_T(coeffs):
    T = np.zeros((2,3))
    T[0, 0] = T[1, 1] = coeffs[0] # sf
    T[0, 2] = coeffs[1] # tx
    T[1, 2] = coeffs[2] # ty
    # T[2, 2] = 1.

    return torch.tensor(T, dtype=torch.float32)

def load_targets(patch_path):
    parent_folder = patch_path.parent
    with open(parent_folder / 'settings.yaml') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    targets = [values for _, values in settings['targets'].items()]
    targets = np.array(targets, dtype=float).T
    targets = torch.from_numpy(targets).float().unsqueeze(0)

    if len(targets.shape) > 2.:
        targets = targets.squeeze(0)
    return targets

def get_coeffs(patch_path, idx=-1):
    parent_folder = patch_path.parent
    T_coeffs = np.load(parent_folder / 'positions_norm.npy') # shape: [sf, tx, ty], epochs, num_targets, 1, 1
    # # should shape: num_targets, [sf, tx, ty]
    T_coeffs = T_coeffs[:, idx, :, 0, 0].T

    return T_coeffs

def get_Ts(T_coeffs):
    Ts = [gen_T(coeffs) for coeffs in T_coeffs]
    return Ts

def calc_loss(target, patch, T, img, model):
    mod_img_pt = place_patch(img.to(patch.device), patch, T.unsqueeze(0).to(patch.device), random_perspection=False) 
    x, y, z, yaw = model(mod_img_pt)
    predicted_pose = torch.hstack((x, y, z))[0].detach().cpu()
    # print(predicted_pose, target)
    mse = torch.nn.functional.mse_loss(target.detach().cpu(), predicted_pose).item()
    return mse

def idx_best_target(targets, patch, Ts, img, model):
    # print("inside best target")
    # print(targets)
    # print(Ts)
    losses = np.array([calc_loss(target, patch, T, img, model) for target, T in zip(targets, Ts)])
    return np.argmin(losses)

def get_best_target_pos(patch, patch_path, dataset, model, device):
    targets = load_targets(patch_path).to(device)
    coeffs = get_coeffs(patch_path)
    Ts = get_Ts(coeffs)

    patch = torch.tensor(patch, device=device).float().unsqueeze(0).unsqueeze(0)

    best_targets = []
    for data in dataset:
        img, _ = data
        best_targets.append(idx_best_target(targets, patch, Ts, img, model))

    idx_best = np.argmax(np.bincount(np.array(best_targets)))

    return targets[idx_best].detach().cpu().numpy(), coeffs[idx_best]


def train(idx_start=0, idx_end=100, model='frontnet'):
    for i in range(idx_start, idx_end):
        with open('dataset.yaml') as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)
        patch_size = settings['patch']['size']
        path = Path(f"{settings['path']}/{patch_size[0]}x{patch_size[1]}/{i}/")
        # print(path)

        targets = [values for _, values in settings['targets'].items()]
        targets = np.array(targets, dtype=float).T

        number_targets = np.random.randint(1, 4)

        random_target_x = np.random.uniform(0,2,number_targets)
        random_target_y = np.random.uniform(-1,1,number_targets,)
        random_target_z = np.random.uniform(-0.5,0.5,number_targets,)

        # overwrite settings
        settings['path'] = str(path)
        settings['targets']['x'] =  random_target_x.tolist()
        settings['targets']['y'] = random_target_y.tolist()
        settings['targets']['z'] = random_target_z.tolist()
        settings['patch']['size'] = patch_size

        os.makedirs(path, exist_ok = True)
        with open(path / 'settings.yaml', 'w') as f:
            yaml.dump(settings, f)

        command = shlex.split(f"sbatch dataset.sh {str(path / 'settings.yaml')} {model}")
        subprocess.run(command)
        del settings

def read_data(path, idx_start=0, idx_end=100, model='frontnet', idx=-1):
    with open('dataset.yaml') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    
    patch_size = settings['patch']['size']
    path = Path(path)

    file_paths = list(path.glob('[0-9]*/patches.npy'))
    file_paths.sort(key=lambda path: int(path.parent.name))

    print(len(file_paths))

    from util import load_dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = "pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"
    dataset = load_dataset(dataset_path, batch_size=1, train=False, train_set_size=0.9)

    if model == 'frontnet':
        from util import load_model
        model_path = 'pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
        model_config = '160x32'
        model = load_model(path=model_path, device=device, config=model_config)
        model.eval()
    else:
        from yolo_bounding import YOLOBox
        model = YOLOBox()
        model.to(device)

    # print(model)
    
    patches = np.array([np.load(file_path)[idx][0][0] for file_path in file_paths[idx_start:idx_end]]) * 255.
    print(patches.shape, patches.min(), patches.max())
    all_images = dataset.dataset.data
    print(all_images.shape)

    targets = [load_targets(patch_path).to(device) for patch_path in file_paths[idx_start:idx_end]]
    print(len(targets))
    
    coeffs = [get_coeffs(patch_path, idx) for patch_path in file_paths[idx_start:idx_end]]

    best_targets = []
    best_coeffs = []
    with torch.no_grad():
        for i, (patch, p_targets, p_coeffs) in enumerate(zip(patches, targets, coeffs)):
            losses = []
            patch = torch.tensor(patch, device=device).float().unsqueeze(0).unsqueeze(0)
            Ts = get_Ts(p_coeffs)
            
            for idx_target in range(len(p_targets)):
                T = Ts[idx_target]
                mod_img_pt = place_patch(all_images.to(patch.device), patch.repeat((len(all_images), 1, 1, 1)), T.unsqueeze(0).repeat(len(all_images), 1, 1).to(patch.device), random_perspection=False) 
                if model == 'frontnet':
                    x, y, z, yaw = model(mod_img_pt)
                    predicted_pose = torch.hstack((x, y, z)).detach().cpu()
                else: 
                    predicted_pose = model(mod_img_pt)
                print(all_images.shape)
                print(predicted_pose.shape)
                print(p_targets, p_targets.shape, p_targets[idx_target], p_targets[idx_target].shape)
                
                # print(p_targets[idx_target].repeat(len(all_images), 1).shape)
                losses.append(torch.nn.functional.mse_loss(p_targets[idx_target].repeat(len(all_images), 1), predicted_pose).detach().cpu().item())
            
            best_targets.append(p_targets[np.argmin(losses)].detach().cpu().numpy())
            best_coeffs.append(p_coeffs[np.argmin(losses)])

    targets = np.array(best_targets)
    positions = np.array(best_coeffs)

    print(targets.shape)
    print(positions.shape)

    return patches, targets, positions
    # out = np.array([get_best_target_pos(patch, path, dataset, model, device) for patch, path in zip(patches, file_paths[idx_start:idx_end+1])])
    # targets, positions = np.split(out, 2, axis=1)
    
def save_pickle(path, out_name, idx_start=0, idx_end=100, model='frontnet'):
    patches, targets, positions = read_data(path, idx_start, idx_end, model)
    with open(out_name, 'wb') as f:
        pickle.dump([[patch/255., target, position] for patch, target, position in zip(patches, targets, positions)], f, pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['train', 'save'])
    parser.add_argument('idx', type=int, metavar='N', nargs='+', default=[100])
    parser.add_argument('--folder', type=str, default='results/finetuning/')
    parser.add_argument('--out', type=str, default='dataset.pickle')
    parser.add_argument('--model', type=str, default='frontnet', choices=['frontnet', 'yolov5'])
    args = parser.parse_args()


    if len(args.idx) < 2:
        idx_start = 0
        idx_end = args.idx[0]
    else:
        idx_start, idx_end = np.sort(args.idx)

    if args.mode == 'train':
        train(idx_start, idx_end, args.model)
    
    if args.mode == 'save':
        save_pickle(args.folder, args.out, idx_start, idx_end, args.model)
