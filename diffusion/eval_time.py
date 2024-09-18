import torch
import numpy as np

from tqdm import trange
from time import time

import yaml
import argparse
import os


from diffusion.diffusion_model import DiffusionModel
# from eval_diffusion import get_alpha_betas, sample

import subprocess

def get_time_fap(n_runs, n_samples):

    all_times = []
    for i in trange(n_runs):
        t_start = time()
        with open('exp_time.yaml') as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)

        model = DiffusionModel(device)
        model.to(device)

        model.load_state_dict(torch.load('diffusion/diffusion_model.pth', map_location=device))
        model.eval()
            
        random_target_x = np.random.uniform(0,2,1)
        random_target_y = np.random.uniform(-1,1,1)
        random_target_z = np.random.uniform(-0.5,0.5,1)

        targets = torch.tensor(np.stack((x, y, z)).T, dtype=torch.float32).repeat(n_samples, 1)
        print(targets.shape)
        
        with torch.no_grad():
                samples = model.sample(n_samples, targets, device, patch_size=[80, 80], n_steps=1_000).detach().cpu().numpy()

        np.save(f'src/custom_patches/time_eval/{n_samples}.npy')

        # overwrite settings
        for idx in range(len(samples)):
            settings['path'] = settings['path'] + f"/ft/n_patches_{n_samples}/{idx}"
            settings['targets']['x'] = random_target_x.item()
            settings['targets']['y'] = random_target_y.item()
            settings['targets']['z'] = random_target_z.item()
            settings['num_patches'] = 1
            settings['patch']['path'] = 'src/custom_patches/time_eval/{n_samples}.npy'



            os.makedirs(settings['path'], exist_ok = True)
            with open(settings['path'] + '/settings.yaml', 'w') as f:
                yaml.dump(settings, f)

            with open(settings['path'] + "/stderr.txt", "w") as stderr:
                with open(settings['path'] + "/stdout.txt", "w") as stdout:
                    subprocess.run(["python3", "src/attacks.py", "--file", settings['path'] + '/settings.yaml'], stderr=stderr, stdout=stdout)
            all_times.append(time()-t_start)

    all_times = np.array(all_times)
    np.save(f'results/time/comp_time_fap_{n_samples}.npy', all_times)

    print("Computation time mean: ", np.mean(all_times), ', std: ', np.std(all_times))

def get_time_fap(n_runs, n_samples):
    all_times = []
    for i in trange(n_runs):
        t_start = time()
        with open('exp_time.yaml') as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)
            
        random_target_x = np.random.uniform(0,2,1)
        random_target_y = np.random.uniform(-1,1,1)
        random_target_z = np.random.uniform(-0.5,0.5,1)

        # overwrite settings
        settings['path'] = settings['path'] + f"/ft/npatches_{n_samples}/{i}"
        settings['targets']['x'] = random_target_x.item()
        settings['targets']['y'] = random_target_y.item()
        settings['targets']['z'] = random_target_z.item()
        settings['num_patches'] = n_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('samples', type=int)
    args = parser.parse_args()

    np.random.seed(2562)
    torch.manual_seed(2562)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_runs = 100

    get_time_fap(n_runs, n_samples=args.samples)