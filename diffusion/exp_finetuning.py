import yaml
from multiprocessing import Pool
import argparse
import copy
import tempfile
from pathlib import Path
import subprocess
import signal
import os
import numpy as np
from collections import defaultdict

import pickle

from time import sleep

def run_attack(settings_path):
    #with tempfile.TemporaryDirectory() as tmpdirname:
    #    filename = Path(tmpdirname) / 'settings.yaml'

    folder = Path(settings_path).parent
    # print(folder)

    job_file = Path(folder) / f"finetuning.sbatch"
    log_file = Path(folder) / "log.out"
    error_file = Path(folder) / "error.out"

    # print(job_file)

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH -c 4\n")
        fh.writelines("#SBATCH --gres=gpu:1\n")
        fh.writelines("#SBATCH -p gpu\n")
        fh.writelines(f"#SBATCH --output={log_file}\n")
        fh.writelines('#SBATCH --time=00:30:00\n\n')
        # fh.writelines(f"#SBATCH --error={error_file}\n\n")
        fh.writelines("source /home/hanfeld/.front-env/bin/activate\n")
        fh.writelines(f"python /home/hanfeld/flying_adversarial_patch/src/attacks.py --file {settings_path}")

    os.system("sbatch %s" %job_file)
    sleep(0.2)

# def get_gt_targets(path, indices):
#     targets = {}
#     for settings_idx in indices:
#         with open(path / str(settings_idx) / 'settings.yaml') as f:
#             settings = yaml.load(f, Loader=yaml.FullLoader)
#         targets[settings_idx] = settings['targets']
#     return targets

def inverse_norm(val, minimum, maximum):
    return np.arctanh(2* ((val - minimum) / (maximum - minimum)) - 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='exp_diffusion.yaml')
    parser.add_argument('--norun', action='store_true')
    parser.add_argument('--quantized', action='store_true')
    parser.add_argument('-j', type=int, default=4)
    parser.add_argument('--hl_iters', type=int, default=10)
    # parser.add_argument('--mode', nargs='+', default=['all']) # mode can be 'all' or ['fixed', 'joint', 'split', 'hybrid']
    args = parser.parse_args()

    # if 'all' in args.mode:
    #     modes = ['fixed', 'joint', 'split', 'hybrid']
    # elif set(args.mode) & set(['fixed', 'joint', 'split', 'hybrid']):
    #     modes = args.mode
    # else:
    #     print("Mode can be either 'all' or a combination from ['fixed', 'joint', 'split', 'hybrid']")


    # SETTINGS
    with open(args.file) as f:
        base_settings = yaml.load(f, Loader=yaml.FullLoader)

    # rewrite settings according to args
    # also read original settings files to get original targets of ground truth patches
    gt_path = Path('diffusion/FAP_combined.pickle')
    gt_indices = np.load('results/fine-tuning80x80/indices.npy') 

    print(gt_indices)
        
    with open(gt_path, 'rb') as f:
        data = pickle.load(f)

    gt_targets = []
    gt_positions = []
    for i in gt_indices:
        gt_targets.append(data[i][1])
        gt_positions.append(data[i][2])

    gt_targets = np.array(gt_targets)
    gt_positions = np.array(gt_positions)

    print(gt_targets[0])
    print(gt_targets[0][0])
    print(gt_targets[0][1])
    print(gt_targets[0][2])

    print(gt_targets[3])
    print(gt_targets[3][0])
    print(gt_targets[3][1])
    print(gt_targets[3][2])

    scale_min = 0.2
    scale_max = 0.7
    tx_min=-10.
    tx_max=100.
    ty_min=-10.
    ty_max=80.

    unnorm_positions = []
    for [sf, tx, ty] in gt_positions:
        sf_unnorm = inverse_norm(sf, scale_min, scale_max)
        tx_unnorm = inverse_norm(tx, tx_min, tx_max)
        ty_unnorm = inverse_norm(ty, ty_min, ty_max)
        unnorm_positions.append([sf_unnorm, tx_unnorm, ty_unnorm])

    unnorm_positions = np.array(unnorm_positions)

    all_settings = []
    base_path = Path(base_settings['path'])
    for patch_idx in range(len(gt_indices)):
        s = copy.copy(base_settings)
        s['path'] = str(base_path / ('hl_iter_' + str(args.hl_iters)) / str(patch_idx))
        os.makedirs(s['path'], exist_ok = True)
        s['targets']['x'] = copy.copy(float(gt_targets[patch_idx][0]))
        s['targets']['y'] = copy.copy(float(gt_targets[patch_idx][1]))
        s['targets']['z'] = copy.copy(float(gt_targets[patch_idx][2]))
        s['patch']['position'] = copy.copy(unnorm_positions[patch_idx].tolist())
        s['num_hl_iter'] = args.hl_iters
        filename = s['path'] + '/settings.yaml'
        # print(filename)
        with open(filename, 'w') as f:
            yaml.dump(s, f)

        all_settings.append(filename)
   
    
    if not args.norun:
        for settings in all_settings:
            print(settings)
            run_attack(settings)



if __name__ == '__main__':
    main()