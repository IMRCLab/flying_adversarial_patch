import numpy as np
import os
import yaml
import argparse
import torch
from torch.nn.functional import mse_loss
from tqdm import trange

from patch_placement import place_patch

from pathlib import Path

def get_transformation(sf, tx, ty):
    translation_vector = torch.stack([tx, ty]).unsqueeze(0) #torch.zeros([1], device=tx.device)]).unsqueeze(0)

    eye = torch.eye(2, 2).unsqueeze(0).to(sf.device)
    scale = eye * sf

    # # print(scale.shape, translation_vector.shape)

    transformation_matrix = torch.cat([scale, translation_vector], dim=2)
    
    # M = torch.eye(3,3).unsqueeze(0).to(sf.device)
    # M[..., :2, :2] *= sf
    # M[..., 0, 2] = tx
    # M[..., 1, 2] = ty
    return transformation_matrix.float()

def norm_transformation(sf, tx, ty, scale_min=0.3, scale_max=0.5, tx_min=-10., tx_max=100., ty_min=-10., ty_max=80.):
    # tx_tanh = torch.tanh(tx) #* 0.8
    # ty_tanh = torch.tanh(ty) #* 0.8

    # new patch placement implementation might need different tx, ty limits!:
    tx_norm = (tx_max - tx_min) * (torch.tanh(tx) + 1) * 0.5 + tx_min
    ty_norm = (ty_max - ty_min) * (torch.tanh(ty) + 1) * 0.5 + ty_min

    scaling_norm = (scale_max - scale_min) * (torch.tanh(sf) + 1) * 0.5 + scale_min # normalizes scaling factor to range [0.3, 0.5]

    return scaling_norm, tx_norm, ty_norm

# def get_rotation(yaw, pitch, roll):
#     rotation_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0.0, 0.0],
#                          [np.sin(yaw), np.cos(yaw), 0.0, 0.0],
#                          [0.0, 0.0, 1.0, 0.0],
#                          [0.0, 0.0, 0.0, 1.0]])
    
#     rotation_pitch = np.array([[np.cos(pitch), 0.0, np.sin(pitch), 0.0],
#                            [0.0, 1.0, 0.0, 0.0],
#                            [-np.sin(pitch), 0.0, np.cos(pitch), 0.0],
#                            [0.0, 0.0, 0.0, 1.0]])

#     rotation_roll = np.array([[1.0, 0.0, 0.0, 0.0],
#                           [0.0, np.cos(roll), -np.sin(roll), 0.0],
#                           [0.0, np.sin(roll), np.cos(roll), 0.0],
#                           [0.0, 0.0, 0.0, 1.0]])     

#     rotation_matrix = rotation_yaw @ rotation_pitch @ rotation_roll
#     return rotation_matrix

def gen_noisy_transformations(batch_size, sf, tx, ty, scale_min=0.3, scale_max=0.5):
    noisy_transformation_matrix = []
    for i in range(batch_size):
        sf_n = sf + np.random.normal(0.0, 0.1)
        tx_n = tx + np.random.normal(0.0, 0.1)
        ty_n = ty + np.random.normal(0.0, 0.1)

        scale_norm, tx_norm, ty_norm = norm_transformation(sf_n, tx_n, ty_n, scale_min, scale_max)
        matrix = get_transformation(scale_norm, tx_norm, ty_norm)

        # random_yaw = np.deg2rad(np.random.normal(-10, 10))
        # random_pitch = np.deg2rad(np.random.normal(-5, 5))
        # random_roll = np.deg2rad(np.random.normal(-5, 5))
        #noisy_rotation = torch.tensor(get_rotation(random_yaw, random_pitch, random_roll)).float().to(matrix.device)

        #matrix[..., :3, :3] = noisy_rotation @ matrix[..., :3, :3]

        noisy_transformation_matrix.append(matrix)
    
    return torch.cat(noisy_transformation_matrix)

def targeted_attack_joint(dataset, patch, model, positions, assignment, targets, lr=3e-2, epochs=10, path="eval/", prob_weight=5, scale_min=0.3, scale_max=0.5, target_offsets = [[0,0,0]], position_offsets=[[0,0,0]], stlc_weights=[1.0]):

    patch_t = patch.clone().requires_grad_(True)
    positions_t = positions.clone().requires_grad_(True)

    opt = torch.optim.Adam([patch_t, positions_t], lr=lr)

    losses = []

    try:
        best_loss = np.inf
        best_stats = None
        best_stats_p = None

        for epoch in range(epochs):

            actual_loss = torch.tensor(0.).to(patch.device)
            stats = np.zeros((len(patch_t), len(targets)))
            stats_p = np.zeros((len(patch_t), len(targets)))

            for _, data in enumerate(dataset):
                batch, _ = data
                batch = batch.to(patch.device) / 255. # limit images to range [0-1]

                target_losses = []
                for target_idx, (position, target) in enumerate(zip(positions_t, targets)):
                    # scale_factor, tx, ty = position
                    # noisy_transformations = gen_noisy_transformations(len(batch), scale_factor, tx, ty)
                    # patch_batch = torch.cat([patch_t for _ in range(len(batch))])

                    # mod_img = place_patch(batch.clone(), patch_batch, noisy_transformations) 

                    # multi-version
                    # generate a transformation matrix of batch_size for each of the num_patches positions
                    # this way, each patch will be placed at it's own optimized position with a bit of noise added
                    # shape is should be (num_patches, batch_size, 2, 3)
                    active_patches = assignment[:,target_idx]
                    stats[np.invert(active_patches), target_idx] = np.inf

                    noisy_transformations = torch.stack([gen_noisy_transformations(len(batch), scale_factor, tx, ty, scale_min, scale_max) for scale_factor, tx, ty in position[active_patches]])
                    # print(noisy_transformations.shape)
                    #patch_batch = torch.cat([patch_t for _ in range(len(batch))])
                    
                    patch_batches = torch.cat([x.repeat(len(batch), 1, 1, 1) for x in patch_t[active_patches]]) # get batch_sized batches of each patch in patches, size should be batch_size*num_patches
                    batch_multi = batch.clone().repeat(len(patch_t[active_patches]), 1, 1, 1)
                    transformations_multi = noisy_transformations.view(len(patch_t[active_patches])*len(batch), 2, 3) # reshape transformation matrices
                    #print(transformations_multi.shape)

                    target_loss = torch.zeros(len(patch_t[active_patches]), device=patch.device)
                    for target_offset, position_offset, weight in zip(target_offsets, position_offsets, stlc_weights):

                        noisy_transformations = torch.stack([gen_noisy_transformations(len(batch), scale_factor + position_offset[0], tx + position_offset[1], ty + position_offset[2]) for scale_factor, tx, ty in position[active_patches]])
                        # print(noisy_transformations.shape)
                        #patch_batch = torch.cat([patch_t for _ in range(len(batch))])
                        
                        patch_batches = torch.cat([x.repeat(len(batch), 1, 1, 1) for x in patch_t[active_patches]]) # get batch_sized batches of each patch in patches, size should be batch_size*num_patches
                        batch_multi = batch.clone().repeat(len(patch_t[active_patches]), 1, 1, 1)
                        transformations_multi = noisy_transformations.view(len(patch_t[active_patches])*len(batch), 2, 3) # reshape transformation matrices
                        # print("before patch placement:")
                        # print("transformations shape: ", transformations_multi.shape)
                        # print("patch batch shape: ", patch_batches.shape)
                        

                        mod_img = place_patch(batch_multi, patch_batches, transformations_multi) 


                        mod_img *= 255. # convert input images back to range [0-255.]

                        # add noise to patch+background
                        mod_img += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(mod_img.shape).to(patch.device)
                        # restrict patch+background to stay in range (0., 255.)
                        mod_img.clamp_(0., 255.)

                        # predict x, y, z, yaw
                        x, y, z, phi = model(mod_img)

                        # target_losses.append(torch.mean(all_l2))
                        # prepare shapes for MSE loss
                        # TODO: improve readbility!
                        pred = torch.stack([x, y, z])
                        pred = pred.squeeze(2).mT

                        # only target x,y and z which were previously chosen, otherwise keep x/y/z to prediction
                        mask = torch.isnan(target)
                        target = torch.where(mask, torch.tensor(0., dtype=torch.float32), target + target_offset)
                        # old
                        # target_batch = (pred * mask) + target
                        # target_losses.append(mse_loss(target_batch, pred))

                        # new
                        pred_v = pred.view(len(patch_t[active_patches]), len(batch), 3) # size : num_patches, batch_size, 3
                        target_batch = (pred_v * mask) + target

                        # target_losses.append(mse_loss(target_batch, pred))

                        #target_losses.append(torch.min(mse_loss(target_batch, pred_v)))
                        target_loss += weight * torch.stack([mse_loss(tar, pred) for tar, pred in zip(target_batch, pred_v)]) # calc mse for each of the predictions of each patch

                    
                    stats[active_patches, target_idx] += target_loss.detach().cpu().numpy()
                    #print(target_loss)
                    # # variant1
                    # target_losses.append(torch.min(target_loss)) # keep only the minimum loss

                    # variant2
                    probabilities = torch.nn.functional.softmin(target_loss * prob_weight, dim=0)
                    stats_p[active_patches, target_idx] += probabilities.detach().cpu().numpy()
                    expectation = probabilities.dot(target_loss)
                    # debug
                    # if target_idx in [0,1]:
                    #     target_losses.append(target_loss[0])
                    # else:
                    #     target_losses.append(target_loss[1])
                    target_losses.append(expectation)

                loss = torch.sum(torch.stack(target_losses))
                actual_loss += loss.clone().detach()

                losses.append(loss.clone().detach())

                opt.zero_grad()
                loss.backward()
                opt.step()

                patch_t.data.clamp_(0., 1.)
            actual_loss /= len(dataset)
            stats /= len(dataset)
            stats_p /= len(dataset)
            print("epoch {} loss {}".format(epoch, actual_loss))
            print("stats loss:", stats)
            print("stats probabilities:", stats_p)
            if actual_loss < best_loss:
                best_patch = patch_t.clone().detach()
                best_position = positions_t.clone().detach()
                best_loss = actual_loss
                best_stats = stats
                best_stats_p = stats_p
        
    except KeyboardInterrupt:
        print("Aborting optimization...")

    return best_patch, best_loss, best_position, best_stats, best_stats_p

def targeted_attack_patch(dataset, patch, model, positions, assignment, targets, lr=3e-2, epochs=10, path="eval/", prob_weight=5, scale_min=0.3, scale_max=0.5):

    patch_t = patch.clone().requires_grad_(True)
    opt = torch.optim.Adam([patch_t], lr=lr)

    #optimized_patches = []
    losses = []

    try:
        best_loss = np.inf
        best_stats = None
        best_stats_p = None

        for epoch in range(epochs):

            actual_loss = torch.tensor(0.).to(patch.device)
            stats = np.zeros((len(patch_t), len(targets)))
            stats_p = np.zeros((len(patch_t), len(targets)))

            for _, data in enumerate(dataset):
                batch, _ = data
                batch = batch.to(patch.device) / 255. # limit images to range [0-1]

                target_losses = []
                for target_idx, (position, target) in enumerate(zip(positions, targets)):
                    # scale_factor, tx, ty = position
                    # noisy_transformations = gen_noisy_transformations(len(batch), scale_factor, tx, ty)
                    # patch_batch = torch.cat([patch_t for _ in range(len(batch))])

                    # mod_img = place_patch(batch.clone(), patch_batch, noisy_transformations) 

                    # multi-version
                    # generate a transformation matrix of batch_size for each of the num_patches positions
                    # this way, each patch will be placed at it's own optimized position with a bit of noise added
                    # shape is should be (num_patches, batch_size, 2, 3)
                    active_patches = assignment[:,target_idx]
                    stats[np.invert(active_patches), target_idx] = np.inf

                    noisy_transformations = torch.stack([gen_noisy_transformations(len(batch), scale_factor, tx, ty, scale_min, scale_max) for scale_factor, tx, ty in position[active_patches]])
                    # print(noisy_transformations.shape)
                    #patch_batch = torch.cat([patch_t for _ in range(len(batch))])
                    
                    patch_batches = torch.cat([x.repeat(len(batch), 1, 1, 1) for x in patch_t[active_patches]]) # get batch_sized batches of each patch in patches, size should be batch_size*num_patches
                    batch_multi = batch.clone().repeat(len(patch_t[active_patches]), 1, 1, 1)
                    transformations_multi = noisy_transformations.view(len(patch_t[active_patches])*len(batch), 2, 3) # reshape transformation matrices
                    #print(transformations_multi.shape)

                    mod_img = place_patch(batch_multi, patch_batches, transformations_multi) 


                    mod_img *= 255. # convert input images back to range [0-255.]

                    # add noise to patch+background
                    mod_img += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(mod_img.shape).to(patch.device)
                    # restrict patch+background to stay in range (0., 255.)
                    mod_img.clamp_(0., 255.)

                    # predict x, y, z, yaw
                    x, y, z, phi = model(mod_img)

                    # prepare shapes for MSE loss
                    #target_batch = target.repeat(len(batch), 1)
                    # TODO: improve readbility!
                    pred = torch.stack([x, y, z])
                    pred = pred.squeeze(2).mT

                    # only target x,y and z which were previously chosen, otherwise keep x/y/z to prediction
                    mask = torch.isnan(target)
                    target = torch.where(mask, torch.tensor(0., dtype=torch.float32), target)

                    # old
                    # target_batch = (pred * mask) + target
                    # target_losses.append(mse_loss(target_batch, pred))

                    # new
                    pred_v = pred.view(len(patch_t[active_patches]), len(batch), 3) # size : num_patches, batch_size, 3
                    target_batch = (pred_v * mask) + target

                     # target_losses.append(mse_loss(target_batch, pred))

                    #target_losses.append(torch.min(mse_loss(target_batch, pred_v)))
                    target_loss = torch.stack([mse_loss(tar, pred) for tar, pred in zip(target_batch, pred_v)]) # calc mse for each of the predictions of each patch
                    stats[active_patches, target_idx] += target_loss.detach().cpu().numpy()
                    #print(target_loss)
                    # # variant1
                    # target_losses.append(torch.min(target_loss)) # keep only the minimum loss

                    # variant2
                    probabilities = torch.nn.functional.softmin(target_loss * prob_weight, dim=0)
                    stats_p[active_patches, target_idx] += probabilities.detach().cpu().numpy()
                    expectation = probabilities.dot(target_loss)
                    # debug
                    # if target_idx in [0,1]:
                    #     target_losses.append(target_loss[0])
                    # else:
                    #     target_losses.append(target_loss[1])
                    target_losses.append(expectation)

                loss = torch.sum(torch.stack(target_losses))
                actual_loss += loss.clone().detach()

                losses.append(loss.clone().detach())

                opt.zero_grad()
                loss.backward()
                opt.step()

                patch_t.data.clamp_(0., 1.)

                #optimized_patches.append(patch_t.clone().detach())
            actual_loss /= len(dataset)
            stats /= len(dataset)
            stats_p /= len(dataset)
            print("epoch {} loss {}".format(epoch, actual_loss))
            print("stats loss:", stats)
            print("stats probabilities:", stats_p)
            if actual_loss < best_loss:
                best_patch = patch_t.clone().detach()
                best_loss = actual_loss
                best_stats = stats
                best_stats_p = stats_p
        
    except KeyboardInterrupt:
        print("Aborting optimization...")    

    #losses = torch.stack(losses)
    return best_patch, best_loss, best_stats, best_stats_p

def targeted_attack_position(dataset, patch, model, target, lr=3e-2, include_start=False, tx_start=0., ty_start=0., sf_start=0.1, num_restarts=50, epochs=5, path="eval/targeted/", scale_min=0.3, scale_max=0.5): 
    try: 
        best_loss = np.inf

        for restart in range(num_restarts):
            if include_start and restart == 0:
                # start with previously optimized values, fine-tuning
                tx = tx_start.clone().to(patch.device).requires_grad_(True)
                ty = ty_start.clone().to(patch.device).requires_grad_(True)
                scaling_factor = sf_start.clone().to(patch.device).requires_grad_(True)
            else:
                # start with random values 
                tx = torch.FloatTensor(1,).uniform_(-1., 1.).to(patch.device).requires_grad_(True)
                ty = torch.FloatTensor(1,).uniform_(-1., 1.).to(patch.device).requires_grad_(True)
                scaling_factor = torch.FloatTensor(1,).uniform_(-1., 1.).to(patch.device).requires_grad_(True)

            
            opt = torch.optim.Adam([scaling_factor, tx, ty], lr=lr)

            mask = torch.isnan(target)
            target = torch.where(mask, torch.tensor(0., dtype=torch.float32), target)
            
            for epoch in range(epochs):

                actual_loss = torch.tensor(0.).to(patch.device)
                for _, data in enumerate(dataset):
                    batch, _ = data
                    batch = batch.to(patch.device) / 255. # limit images to range [0-1]
            
                    noisy_transformations = gen_noisy_transformations(len(batch), scaling_factor, tx, ty, scale_min, scale_max)
                    patch_batch = torch.cat([patch for _ in range(len(batch))])
 
                    mod_img = place_patch(batch.clone(), patch_batch, noisy_transformations)
                    mod_img *= 255.  # convert input images back to range [0-255.]
                    # add noise to patch+background
                    mod_img += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(mod_img.shape).to(patch.device)    
                    mod_img.clamp_(0., 255.)

                    x, y, z, phi = model(mod_img)

                    # calculate mean l2 losses (target, prediction) for all images in batch
                    # prepare shapes for MSE loss
                    # TODO: improve readbility!
                    pred = torch.stack([x, y, z])
                    pred = pred.squeeze(2).mT

                    # only target x,y and z which were previously chosen, otherwise keep x/y/z to prediction
                    #torch.where(torch.isnan(target_batch), pred, target_batch)
                    target_batch = (pred * mask) + target

                    loss = mse_loss(target_batch, pred)

                    actual_loss += loss.clone().detach()

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                actual_loss /= len(dataset)
                if actual_loss < best_loss:
                    print("restart {} improved loss to {}".format(restart, actual_loss))
                    best_loss = actual_loss
                    best_tx = tx.clone().detach()
                    best_ty = ty.clone().detach()
                    best_scaling = scaling_factor.clone().detach()

    except KeyboardInterrupt:
        print("Aborting optimization...")

    return best_scaling, best_tx, best_ty, best_loss

def calc_eval_loss(dataset, patch, transformation_matrix, model, target, quantized=False):
    actual_loss = torch.tensor(0.).to(patch.device)

    mask = torch.isnan(target)
    target = torch.where(mask, torch.tensor(0., dtype=torch.float32), target)

    for _, data in enumerate(dataset):
        batch, _ = data
        batch = batch.to(patch.device) / 255. # limit images to range [0-1]

        mod_img = place_patch(batch, patch, transformation_matrix)
        mod_img *= 255. # convert input images back to range [0-255.]
        mod_img.clamp_(0., 255.)
        if quantized:
            mod_img.floor_()

        x, y, z, phi = model(mod_img)

        # prepare shapes for MSE loss
        # target_batch = target.repeat(len(batch), 1)
        pred = torch.stack([x, y, z])
        pred = pred.squeeze(2).mT

        # only target x,y and z which are previously chosen, otherwise keep x/y/z to prediction
        #target_batch = torch.where(torch.isnan(target_batch), pred, target_batch)
        target_batch = (pred * mask) + target
        
        loss = mse_loss(target_batch, pred)
        actual_loss += loss.clone().detach()
    
    actual_loss /= len(dataset)

    return actual_loss



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='settings.yaml')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    # SETTINGS
    with open(args.file) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    # save settings directly to result folder for later use
    path = Path(settings['path'])
    os.makedirs(path, exist_ok = True)
    with open(path / 'settings.yaml', 'w') as f:
        yaml.dump(settings, f)

    lr_pos = settings['lr_pos']
    lr_patch = settings['lr_patch']
    num_hl_iter = settings['num_hl_iter']
    num_pos_restarts = settings['num_pos_restarts']
    num_pos_epochs = settings['num_pos_epochs']
    num_patch_epochs = settings['num_patch_epochs']
    batch_size = settings['batch_size']
    mode = settings['mode']
    quantized = settings['quantized']
    prob_weight = settings['prob_weight']
    scale_min = settings['scale_min']
    scale_max = settings['scale_max']

    stlc_target_offsets = torch.tensor([o['target'] for o in settings['stlc']['offsets']], dtype=torch.float, device=device)
    stlc_position_offsets = [o['position'] for o in settings['stlc']['offsets']]
    stlc_weights = [o['weight'] if 'weight' in o else 1.0 for o in settings['stlc']['offsets']]

    # get target values in correct shape and move tensor to device
    targets = [values for _, values in settings['targets'].items()]
    targets = np.array(targets, dtype=float).T
    targets = torch.from_numpy(targets).to(device).float()

    from util import load_dataset
    dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'

    # model = load_model(path=model_path, device=device, config=model_config)
    # model.eval()
    print("Loading quantized network? ", quantized)
    if not quantized:
        # load full-precision network
        from util import load_model
        model_path = 'pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
        model_config = '160x32'
        model = load_model(path=model_path, device=device, config=model_config)
    else:
        # load quantized network
        from util import load_quantized
        model_path = 'misc/Frontnet.onnx'
        model = load_quantized(path=model_path, device=device)
    
    model.eval()

    train_set = load_dataset(path=dataset_path, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    # train_set.dataset.data.to(device)   # TODO: __getitem__ and next(iter(.)) are still yielding data on cpu!
    # train_set.dataset.labels.to(device)
    
    test_set = load_dataset(path=dataset_path, batch_size=batch_size, shuffle=True, drop_last=False, train=False, num_workers=0)

    num_patches = settings['num_patches']

    # load the patch from misc folder
    if settings['patch']['mode'] == 'face':
        patch_start = np.load(settings['patch']['path'])
        patch_start = torch.from_numpy(patch_start).unsqueeze(0).unsqueeze(0).to(device) / 255.
        patch_start.clamp_(0., 1.)
        patch_start = torch.stack([patch_start[0].clone() for _ in range(num_patches)])

    # or start from a random patch
    if settings['patch']['mode'] == 'random':
        size = settings['patch']['size']
        patch_start = torch.rand(num_patches, 1, size[0], size[1]).to(device)

    # or start from a white patch
    if settings['patch']['mode'] == 'white':
        size = settings['patch']['size']
        patch_start = torch.ones(num_patches, 1, size[0], size[1]).to(device)

    optimization_pos_losses = []
    optimization_pos_vectors = []

    optimization_patches = []
    optimization_patch_losses = []

    train_losses = []
    test_losses = []

    stats_all = []
    stats_p_all = []

    positions = torch.FloatTensor(len(targets), num_patches, 3, 1).uniform_(-1., 1.).to(device)
    # sf = torch.FloatTensor(len(targets), num_patches, 1).uniform_(-1, 1.).to(device)
    # tx = torch.FloatTensor(len(targets), num_patches, 1).uniform_(-1, 1.).to(device)
    # ty = torch.FloatTensor(len(targets), num_patches, 1).uniform_(-1, 1.).to(device)
    # positions = torch.stack([sf, tx, ty]).moveaxis(0, 2)

    optimization_pos_vectors.append(positions)

    # num_patches x bitness x width x height
    patch = patch_start.clone()
    optimization_patches.append(patch.clone())

    # assignment: we start by assigning all targets to all patches
    A = np.ones((num_patches, len(targets)), dtype=np.bool_)

    # # debug
    # A = np.zeros((num_patches, len(targets)), dtype=np.bool8)
    # A[0,0:2] = True
    # A[1,2:4] = True
    # print(A)
    
    for train_iteration in trange(num_hl_iter):
        
        pos_losses = []

        if mode == "split" or mode == "fixed":
            print("Optimizing patch...")
            patch, loss_patch, stats, stats_p = targeted_attack_patch(train_set, patch, model, optimization_pos_vectors[-1], A, targets=targets, lr=lr_patch, epochs=num_patch_epochs, path=path, prob_weight=prob_weight, scale_min=scale_min, scale_max=scale_max)
            stats_all.append(stats)
            stats_p_all.append(stats_p)
        elif mode == "joint" or mode == "hybrid":
            patch, loss_patch, positions, stats, stats_p = targeted_attack_joint(train_set, patch, model, optimization_pos_vectors[-1], A, targets=targets, lr=lr_patch, epochs=num_patch_epochs, path=path, prob_weight=prob_weight, scale_min=scale_min, scale_max=scale_max, target_offsets=stlc_target_offsets, position_offsets=stlc_position_offsets,
            stlc_weights=stlc_weights)
            optimization_pos_vectors.append(positions)

            pos_losses.append(loss_patch)
            stats_all.append(stats)
            stats_p_all.append(stats_p)

        optimization_patches.append(patch.clone())
        optimization_patch_losses.append(loss_patch)

        if mode == "split" or mode == "hybrid":
            # optimize positions for multiple target values
            positions = torch.empty(len(targets), num_patches, 3, 1, device=device)
            for target_idx, target in enumerate(targets):
                for patch_idx in range(num_patches):
                    scale_start, tx_start, ty_start = optimization_pos_vectors[-1][target_idx][patch_idx]
                    if A[patch_idx, target_idx]:
                        print(f"Optimizing position for patch {patch_idx} and target {target.cpu().numpy()}...")
                        scale_factor, tx, ty, loss_pos  = targeted_attack_position(train_set, patch[patch_idx:patch_idx+1], model, target, include_start=True, tx_start=tx_start, ty_start=ty_start, sf_start=scale_start, lr=lr_pos, num_restarts=num_pos_restarts, epochs=num_pos_epochs, path=path, scale_min=scale_min, scale_max=scale_max)
                        positions[target_idx, patch_idx] = torch.stack([scale_factor, tx, ty])
                        pos_losses.append(loss_pos)
                    else:
                        positions[target_idx, patch_idx] = torch.stack([scale_start, tx_start, ty_start])
                        pos_losses.append(torch.tensor(np.inf, device=device))
        elif mode == "fixed":
            pos_losses = [torch.tensor([0.])]

        optimization_pos_vectors.append(positions)
        optimization_pos_losses.append(torch.stack(pos_losses))
        print(optimization_pos_vectors[-1].shape)

        train_loss = []
        test_loss = []
        cost = np.zeros((num_patches, len(targets)))
        for target_idx, target in enumerate(targets):
            train_losses_per_patch = []
            test_losses_per_patch = []
            for patch_idx in range(num_patches):
                scale_norm, tx_norm, ty_norm = norm_transformation(*optimization_pos_vectors[-1][target_idx][patch_idx], scale_min, scale_max)
                transformation_matrix = get_transformation(scale_norm, tx_norm, ty_norm).to(device)

                train_losses_per_patch.append(calc_eval_loss(train_set, patch[patch_idx:patch_idx+1], transformation_matrix, model, target, quantized=quantized))
                test_losses_per_patch.append(calc_eval_loss(test_set, patch[patch_idx:patch_idx+1], transformation_matrix, model, target, quantized=quantized))
                cost[patch_idx, target_idx] = test_losses_per_patch[-1]
            # only store the best loss per target
            train_loss.append(torch.min(torch.as_tensor(train_losses_per_patch)))
            test_loss.append(torch.min(torch.as_tensor(test_losses_per_patch)))

        train_losses.append(torch.stack(train_loss))
        test_losses.append(torch.stack(test_loss))

        if mode == "split" or mode == "hybrid":
            print("Assignment")
            print("cost", cost)
            # optimal assignment, since we have a 1:n matching, we can just assign each target the patch with the lowest loss
            # variant 1: deterministic assignment
            A = cost == np.min(cost, axis=0)
            # variant 2: stochastic assignment
            probabilities = torch.nn.functional.softmin(prob_weight * torch.as_tensor(cost), dim=0)
            print("P", probabilities)
            sample = torch.multinomial(probabilities.T, 1).flatten()
            print("sample", sample)
            A[:,:] = False
            for target_idx in range(len(targets)):
                A[sample[target_idx], target_idx] = True
            print("A", A)

    #print(optimization_patch_losses)
    optimization_patches = torch.stack(optimization_patches)
    optimization_patch_losses = torch.stack(optimization_patch_losses)
    
    #print(optimization_pos_vectors)
    optimization_pos_vectors = torch.stack(optimization_pos_vectors)
    optimization_pos_losses = torch.stack(optimization_pos_losses)
    

    train_losses = torch.stack(train_losses)
    test_losses = torch.stack(test_losses)

    print("Saving results...")
    # prepare data for plots
    # normalize scale factor, tx and ty for plots

    norm_optimized_vecs = [norm_transformation(optimization_pos_vectors[i].mT[..., 0], optimization_pos_vectors[i].mT[..., 1], optimization_pos_vectors[i].mT[..., 2], scale_min, scale_max) for i in range(len(optimization_pos_vectors))]

    all_sf = torch.stack([norm_optimized_vecs[i][0] for i in range(len(norm_optimized_vecs))])
    all_tx = torch.stack([norm_optimized_vecs[i][1] for i in range(len(norm_optimized_vecs))])
    all_ty = torch.stack([norm_optimized_vecs[i][2] for i in range(len(norm_optimized_vecs))])


    # save all results in numpy arrays for later use
    np.save(path / 'patches.npy', optimization_patches.cpu().numpy())
    np.save(path / 'patch_losses.npy', optimization_patch_losses.cpu().numpy())
    # np.save(path / 'positions.npy', optimization_pos_vectors.cpu().numpy())
    np.save(path / 'positions_norm.npy', np.array([all_sf.cpu().numpy(), all_tx.cpu().numpy(), all_ty.cpu().numpy()]))
    np.save(path / 'position_losses.npy', optimization_pos_losses.cpu().numpy())
    np.save(path / 'losses_train.npy', train_losses.cpu().numpy())
    np.save(path / 'losses_test.npy', test_losses.cpu().numpy())

    np.save(path / 'stats.npy', stats_all)
    np.save(path / 'stats_p.npy', stats_p_all)

    # final evaluation on test set
    print("Evaluation...")

    test_batch, test_gt = test_set.dataset[:]
    test_batch = test_batch.to(device) / 255. # limit images to range [0-1]

    boxplot_data = []
    #target_mask = torch.tensor(target_mask).to(patch.device)
    for target_idx, target in enumerate(targets):
        pred_base = model(test_batch.float() * 255.)
        pred_base = torch.stack(pred_base[:3]).squeeze(2).mT
        target_batch = target.repeat(len(test_batch), 1)
        target_batch = torch.where(torch.isnan(target_batch), pred_base, target_batch)
        loss_base = torch.tensor([mse_loss(target_batch[i], pred_base[i]) for i in range(len(test_batch))])

        loss_start_patch_best = None
        loss_start_patch_best_value = np.inf
        loss_opt_patch_best = None
        loss_opt_patch_best_value = np.inf
        for patch_idx in range(num_patches):
            scale_norm, tx_norm, ty_norm = norm_transformation(*optimization_pos_vectors[-1][target_idx][patch_idx], scale_min, scale_max)
            transformation_matrix = get_transformation(scale_norm, tx_norm, ty_norm).to(device)

            mod_img = place_patch(test_batch, patch_start[patch_idx:patch_idx+1], transformation_matrix)
            mod_img *= 255. # convert input images back to range [0-255.]
            mod_img.clamp_(0., 255.)
            pred_start_patch = model(mod_img.float())
            pred_start_patch = torch.stack(pred_start_patch[:3]).squeeze(2).mT
            target_batch = target.repeat(len(test_batch), 1)
            target_batch = torch.where(torch.isnan(target_batch), pred_start_patch, target_batch)
            loss_start_patch = torch.tensor([mse_loss(target_batch[i], pred_start_patch[i]) for i in range(len(test_batch))])
            if torch.sum(loss_start_patch) < loss_start_patch_best_value:
                loss_start_patch_best_value = torch.sum(loss_start_patch)
                loss_start_patch_best = loss_start_patch

            mod_img = place_patch(test_batch, patch[patch_idx:patch_idx+1], transformation_matrix)
            mod_img *= 255. # convert input images back to range [0-255.]
            mod_img.clamp_(0., 255.)
            pred_opt_patch = model(mod_img.float())
            pred_opt_patch = torch.stack(pred_opt_patch[:3]).squeeze(2).mT
            target_batch = target.repeat(len(test_batch), 1)
            target_batch = torch.where(torch.isnan(target_batch), pred_opt_patch, target_batch)
            loss_opt_patch = torch.tensor([mse_loss(target_batch[i], pred_opt_patch[i]) for i in range(len(test_batch))])
            if torch.sum(loss_opt_patch) < loss_opt_patch_best_value:
                loss_opt_patch_best_value = torch.sum(loss_opt_patch)
                loss_opt_patch_best = loss_opt_patch

        boxplot_data.append(torch.stack([loss_base.detach().cpu(), loss_start_patch_best.detach().cpu(), loss_opt_patch_best.detach().cpu()]))


    np.save(path / 'boxplot_data.npy', torch.stack(boxplot_data).cpu().numpy())

    # STLC Analysis
    for target_idx, target in enumerate(targets):
        for patch_idx in range(num_patches):

            scale_norm, tx_norm, ty_norm = norm_transformation(*optimization_pos_vectors[-1][target_idx][patch_idx], scale_min, scale_max)
            for target_offset, position_offset in zip(stlc_target_offsets, stlc_position_offsets):
                transformation_matrix = get_transformation(scale_norm + position_offset[0], tx_norm + position_offset[1], ty_norm + position_offset[2]).to(device)

                mod_img = place_patch(test_batch, patch[patch_idx:patch_idx+1], transformation_matrix)
                mod_img *= 255. # convert input images back to range [0-255.]
                mod_img.clamp_(0., 255.)
                pred_opt_patch = model(mod_img.float())
                pred_opt_patch = torch.stack(pred_opt_patch[:3]).squeeze(2).mT
                target_batch = (target + target_offset).repeat(len(test_batch), 1)
                target_batch = torch.where(torch.isnan(target_batch), pred_opt_patch, target_batch)
                loss_opt_patch = torch.tensor([mse_loss(target_batch[i], pred_opt_patch[i]) for i in range(len(test_batch))])
                print("STLC", target_offset.detach().cpu().numpy(), torch.mean(loss_opt_patch).detach().cpu().numpy())

    from plots import plot_results
    plot_results(path)
