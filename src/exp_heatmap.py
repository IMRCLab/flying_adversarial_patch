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
    sleep(0.2)

def inverse_norm(val, minimum, maximum):
    return np.arctanh(2* ((val - minimum) / (maximum - minimum)) - 1)

def main():
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

    step_size = 0.1
    y_vals = np.arange(-1, 1 + step_size, step=step_size)
    z_vals = np.arange(-0.5, 0.5 + step_size, step=step_size)

    all_settings = []
    idx = 0
    for y in y_vals:
        for z in z_vals:
            s = copy.copy(base_settings)
            s['path'] = str(base_path / str(idx))
            os.makedirs(s['path'], exist_ok = True)
            s['targets']['y'] = copy.copy([round(float(y), 2)])
            s['targets']['z'] = copy.copy([round(float(z), 2)])
            filename = s['path'] + '/settings.yaml'
            with open(filename, 'w') as f:
                yaml.dump(s, f)
            
            all_settings.append(filename)
            idx += 1

    # print(len(all_settings))

    if not args.norun:
        for settings in all_settings[:2]:
            # print(settings)
            run_attack(settings)

if __name__ == '__main__':
    main()