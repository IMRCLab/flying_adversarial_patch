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

from time import sleep, time

def run_attack(settings_path):
    #with tempfile.TemporaryDirectory() as tmpdirname:
    #    filename = Path(tmpdirname) / 'settings.yaml'

    folder = Path(settings_path).parent
    # print(folder)

    job_file = Path(folder) / f"heatmap.sbatch"
    log_file = Path(folder) / "log.out"
    error_file = Path(folder) / "error.out"

    # print(job_file)

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH -c 6\n")
        fh.writelines("#SBATCH --gres=gpu:1\n")
        fh.writelines("#SBATCH -p gpu_p100\n")
        fh.writelines(f"#SBATCH --output={log_file}\n")
        fh.writelines('#SBATCH --time=01:00:00\n\n')
        # fh.writelines(f"#SBATCH --error={error_file}\n\n")
        fh.writelines("source /home/hanfeld/.front-env/bin/activate\n")
        fh.writelines(f"python /home/hanfeld/flying_adversarial_patch/src/attacks.py --file {settings_path}")

    os.system("sbatch %s" %job_file)
    # sleep(0.2)

def inverse_norm(val, minimum, maximum):
    return np.arctanh(2* ((val - minimum) / (maximum - minimum)) - 1)

def main():
    time_start = time()
    import torch
    np.random.seed(2562)
    torch.manual_seed(2562)
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='exp_diffusion.yaml')
    parser.add_argument('--norun', action='store_true')
    parser.add_argument('--diffusion', action='store_true')
    parser.add_argument('-j', type=int, default=4)
    # parser.add_argument('--hl_iters', type=int, default=10)
    # parser.add_argument('--mode', nargs='+', default=['all']) # mode can be 'all' or ['fixed', 'joint', 'split', 'hybrid']
    args = parser.parse_args()

    # SETTINGS
    with open(args.file) as f:
        base_settings = yaml.load(f, Loader=yaml.FullLoader)

    base_path = Path(base_settings['path'])

    scale_min = base_settings['scale_min']
    scale_max = base_settings['scale_max']
    # tx_min = base_settings['tx_min']
    # tx_max = base_settings['tx_max']
    # ty_min = base_settings['ty_min']
    # ty_max = base_settings['ty_max']
    

    step_size = 0.1
    y_vals = np.arange(-1, 1 + step_size, step=step_size)
    z_vals = np.arange(-0.5, 0.5 + step_size, step=step_size)

    if args.diffusion:
        # load all modules for calculating anytime loss directly after sampling the patch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from util import load_dataset
        dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
        test_set = load_dataset(path=dataset_path, batch_size=base_settings['batch_size'], shuffle=True, drop_last=False, train=False, num_workers=0)
        from util import load_model
        model_path = 'pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
        model_config = '160x32'
        model = load_model(path=model_path, device=device, config=model_config)
        
        from attacks import calc_anytime_loss
    
        from diffusion.diffusion_model import DiffusionModel
        
    load_time = time()-time_start
    print(f"took {load_time} s!")

    for i in range(10):
        all_settings = []
        idx = 0
        for y in y_vals:
            for z in z_vals:
                time_start = time()
                
                
                s = copy.copy(base_settings)


                s['targets']['y'] = copy.copy([round(float(y), 2)])
                s['targets']['z'] = copy.copy([round(float(z), 2)])

                if args.diffusion:
                    dif_model = DiffusionModel(device)
                    dif_model.load('diffusion/diffusion_model.pth')
                    s['patch']['mode'] = 'diffusion'
                
                    # sf = np.random.uniform(scale_min, scale_max)
                    # sf_unnorm = inverse_norm(sf, scale_min, scale_max)
                    # tx = np.random.uniform(tx_min, tx_max)
                    # tx_unnorm = inverse_norm(tx, tx_min, tx_max)
                    # ty = np.random.uniform(ty_min, ty_max)
                    # ty_unnorm = inverse_norm(ty, ty_min, ty_max)
                    
                    # position = np.array([sf_unnorm, tx_unnorm, ty_unnorm])
                    # print(position)

                    position = np.random.uniform(-1., 1., 3)

                    s['patch']['position'] = copy.copy(position.tolist())
                    s['path'] = str(base_path / str(i) / 'diffusion' / str(idx))
                    os.makedirs(s['path'], exist_ok = True)
                    s['patch']['path'] = s['path'] + '/diffusion_patch.npy'
                    target = np.array([values for _, values in s['targets'].items()]).T
                    target = torch.tensor(target, device=device, dtype=torch.float32)

                    patch = dif_model.sample(1, target, device, patch_size=s['patch']['size'], n_steps=1_000).to(device) * 255.
                    np.save(s['patch']['path'], patch[0].cpu().numpy())
                    position_t = torch.tensor(position).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(device)
                    seconds, test_loss = calc_anytime_loss(time_start, test_set, patch, target, model, position_t, scale_min=scale_min, scale_max=scale_max, quantized=False)
                    print(seconds, test_loss)
                    # sanity check
                    # noise = torch.rand_like(patch, device=device, dtype=patch.dtype) * 255.
                    # seconds, test_loss = calc_anytime_loss(time_start, test_set, patch, target, model, position_t, scale_min=scale_min, scale_max=scale_max, quantized=False)
                    # print(seconds, test_loss)
                    seconds += load_time
                    np.save(s['path'] + '/anytime_loss.npy', np.array([[0., np.inf], [seconds, test_loss]]))
                else:
                    s['path'] = str(base_path / 'gt' / str(idx))
                    os.makedirs(s['path'], exist_ok = True)

                filename = s['path'] + '/settings.yaml'
                with open(filename, 'w') as f:
                    yaml.dump(s, f)
                
                all_settings.append(filename)
                idx += 1
                run_attack(filename)
            

    # print(len(all_settings))

    # if not args.norun:
    #     for settings in all_settings[:2]:
            # print(settings)
            # run_attack(settings)

if __name__ == '__main__':
    main()