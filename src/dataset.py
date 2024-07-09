import numpy as np
from pathlib import Path
import torch
import yaml
from cf_frontnet_pt import CFSim
import pickle

def get_targets(task_id):
    with open(f'results/dataset/settings_{task_id}.yaml') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    targets = [values for _, values in settings['targets'].items()]
    targets = np.array(targets, dtype=float).T
    targets = torch.from_numpy(targets).float()
    return targets

def gen_T(coeffs):
    T = np.zeros((3,3))
    T[0, 0] = T[1, 1] = coeffs[0] # sf
    T[0, 2] = coeffs[1] # tx
    T[1, 2] = coeffs[2] # ty
    T[2, 2] = 1.

    return T

def get_Ts(T_coeffs):
    Ts = [gen_T(coeffs) for coeffs in T_coeffs]
    return Ts

def get_pos(path):
    T_coeffs = np.load(path) # shape: 3, num_targets, 1, 1
    # print("t coeffs", T_coeffs)
    # # should shape: num_targets, [sf, tx, ty]
    T_coeffs = T_coeffs[:, :, 0, 0].T
    return T_coeffs

def _calc_loss(target, patch, T, img, sim):
    mod_img_pt = sim.pt_project_patch(patch, T, img) 
    x, y, z, yaw = sim.pose_estimator(mod_img_pt)
    predicted_pose = torch.hstack((x, y, z))[0].detach().cpu()

    mse = torch.nn.functional.mse_loss(target, predicted_pose).item()
    return mse

def loss_dataset(patch, targets, Ts, dataset, sim):
    # losses = np.zeros((np.shape(targets)[0],))
    losses = []
    num_targets = np.shape(targets)[0]
    
    for i in range(num_targets):
        total_loss = np.sum([_calc_loss(targets[i], patch, Ts[i], img, sim) for img, _ in dataset])
        losses.append(total_loss)

    return np.array(losses)

def test_file():
    with open('data.pkl', 'rb') as f:
        patch_info = pickle.load(f) # deserialize using load()
        print(patch_info) # print student names

def main():
    task_id = 0
    max_task = 757

    dataset_path = "pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"
    sim = CFSim(dataset_path=dataset_path)

    _, dataset = sim.load_dataset(dataset_path, batch_size=1, train_set_size=0.9)


    # pickle object: (num_patches, (80, 80), (1, 3), (1, 3))
    all_data = []
    for i in range(task_id, max_task):
        patch_path = Path(f'results/dataset/last_patch_{i}.npy')
        pos_path = Path(f'results/dataset/position_norm_{i}.npy')

        if not (patch_path.exists() and pos_path.exists()):
            print(f'task {i} is missing files')
            continue

        print("=======")
        print("task id", i)
        last_patch = np.load(patch_path)
        last_patch = last_patch[0,0,:,:]
        print('patch shape:', np.shape(last_patch))

        targets = get_targets(i).clone().detach()
        print("targets:", np.shape(targets))
        pos = get_pos(pos_path)
        Ts = get_Ts(pos)
        print("Ts:", np.shape(Ts))

        losses = loss_dataset(last_patch, targets, Ts, dataset, sim)
        best_target = np.argmin(losses)
        print('losses')
        print(np.shape(losses), losses, best_target)

        all_data.append([last_patch, targets[best_target], pos[best_target]])

    # print(all_data[0])
    file = open('data.pkl', 'wb')
    pickle.dump(all_data, file)

    print("dumped")

if __name__ == "__main__":
    main()