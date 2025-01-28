import torch
import numpy as np
import cv2
import os
import glob
import yaml

import pickle

import matplotlib.pyplot as plt

from tqdm import tqdm, trange

from patch_placement import place_patch

# taken from ultralytics yolo
def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

# scale box to original image scale
def scale_box(box, scale_factor_width, scale_factor_height):
    box[0] /= scale_factor_width
    box[1] /= scale_factor_height
    box[2] /= scale_factor_width
    box[3] /= scale_factor_height
    return box

SOFTMAX_SCALE = 20.

def create_pkl(path, file_type='jpg'):
    # load all images from images_folder
    all_images = []
    for file in tqdm(sorted(glob.glob(f'{path}/*.{file_type}'))):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 320))
        #print(img.shape)
        all_images.append(img.transpose(2, 0, 1)) # shape: (3, 320, 640)

    all_images = np.array(all_images)
    with open('misc/dji/all_resized_images.pkl', 'wb') as f:
        pickle.dump(all_images, f)

def inverse_norm(val, minimum, maximum):
    #return torch.arctanh(2* ((val - minimum) / (maximum - minimum)) - 1)
    #return (maximum - minimum) * (torch.tanh(val) + 1) * 0.5 + minimum
    return (maximum - minimum) * (val + 1) * 0.5 + minimum


def get_transformation(sf, tx, ty):
    translation_vector = torch.stack([tx, ty]) # shape (2, 1)
    # print(translation_vector.shape)
    eye = torch.eye(2,2).to(device)
    scale = eye * sf  # shape (2,2)

    # print(scale.shape)

    transformation_matrix = torch.cat([scale, translation_vector], dim=1)
    return transformation_matrix


def gen_noisy_transformations(batch_size, sf, tx, ty, scale_min=0.0, scale_max=1.8, tx_min=0., tx_max=640., ty_min=0., ty_max=320.):
    noisy_transformation_matrix = []
    for i in range(batch_size):
        sf_n = sf + np.random.normal(0.0, 0.1)
        tx_n = tx + np.random.normal(0.0, 0.1)
        ty_n = ty + np.random.normal(0.0, 0.1)

        sf_unnorm = inverse_norm(sf_n, scale_min, scale_max)
        tx_unnorm = inverse_norm(tx_n, tx_min, tx_max)
        ty_unnorm = inverse_norm(ty_n, ty_min, ty_max)

        matrix = get_transformation(sf_unnorm, tx_unnorm, ty_unnorm)

        noisy_transformation_matrix.append(matrix)
    
    return torch.stack(noisy_transformation_matrix)


def xyz_from_bb(bb, camera_intrinsic, radius =0.5):
        device = bb.device
        fx = camera_intrinsic[0, 0]
        fy = camera_intrinsic[1, 1]
        ox = camera_intrinsic[0, 2]
        oy = camera_intrinsic[1, 2]

        # printd('bounding box', bb.grad_fn)
        center = (bb[1] + bb[3])/2

        # get rays for pixels
        a1 = torch.ones(3, device=device)
        a1[0] = (bb[0]-ox)/fx
        a1[1] = (center-oy)/fy

        a2 = torch.ones(3, device=device)
        a2[0] = (bb[2]-ox)/fx
        a2[1] = (center-oy)/fy

        # printd('a1', a1.grad_fn)

        # normalize rays
        a1_norm = torch.linalg.norm(a1)
        a2_norm = torch.linalg.norm(a2)

        # printd('a1 nnorm', a1_norm.grad_fn)

        # get the distance    
        distance = (np.sqrt(2)*radius)/(torch.sqrt(1-torch.dot(a1,a2)/(a1_norm*a2_norm)))

        # printd('distance', distance.grad_fn)

        # get central ray
        ac = (a1+a2)/2

        # get the position
        xyz = distance*ac/torch.linalg.norm(ac)

        #new_xyz = (torch.linalg.inv(camera_extrinsic_tens) @ torch.cat((xyz, torch.ones(1))))[:3]
        return xyz

if __name__ == '__main__':
  
  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load yolov5
    model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True, autoshape=False)
    model.eval()

    scale_factor_height = 320 / 2160
    scale_factor_width = 640 / 3840

    tx_min = 0.
    tx_max = 640.
    ty_min = 0.
    ty_max = 320.
    scale_min = 0.0
    scale_max = 1.8

    # load image from path with cv2
    images_folder = 'misc/dji/images'
    output_folder = 'misc/dji/temp_batch_prediction'
    os.makedirs(output_folder, exist_ok=True)

    patch = torch.rand(1, 3, 80, 80).to(device).requires_grad_(True)
    # print(random_patch.min(), random_patch.max())

    # load camera intrinsic from yaml
    with open('misc/dji/calibration/calibration_scaled.yaml', 'r') as f:
        config = yaml.safe_load(f)

    camera_intrinsic = torch.tensor(config['camera_matrix']).to(device)
    print(camera_intrinsic)
    

    #target_box = torch.tensor([480, 125, 619, 241]).to(device)
    target_position = torch.tensor([0.0, 0.0, 1.5]).to(device) # predicting person 1.5 meters away from camera
    print(target_position.shape)
    target_in_image = camera_intrinsic @ target_position
    target_x = int(target_in_image[0] / target_in_image[2])
    target_y = int(target_in_image[1] / target_in_image[2])
    print(target_x, target_y)

    # create_pkl(images_folder)
    #plt.imsave('misc/dji/temp_load.jpg', all_images[0].transpose(1, 2, 0))


    # load images from pickle
    with open('misc/dji/all_resized_images.pkl', 'rb') as f:
        images = np.array(pickle.load(f))   # shape: (num_images, 3, 320, 640)
    
    # print(images.shape)
    # print(images[0].min(), images[0].max())
    # # permute to (num_images, 3, 320, 640)
    # images = images.transpose(0, 3, 1, 2)
    # print(images.shape)
    images = torch.tensor(images/255.).float()
    # print(images.shape)
    # print(images[0].min(), images[0].max())

    # # plot image with matplotlib
    # plt.imsave('misc/dji/temp.jpg', images[0].permute(1, 2, 0).detach().cpu().numpy())

    dataloader = torch.utils.data.DataLoader(images, batch_size=32, shuffle=True)

    # init random scale_factor, tx, ty in [-1, 1]
    scale_factor = torch.FloatTensor(1,).uniform_(-1., 1.).to(device).requires_grad_(True)
    tx = torch.FloatTensor(1,).uniform_(-1., 1.).to(device).requires_grad_(True)
    ty = torch.FloatTensor(1,).uniform_(-1., 1.).to(device).requires_grad_(True)


    opt = torch.optim.Adam([patch, scale_factor, tx, ty], lr=3e-3)

    best_patch = None
    best_position = None
    best_loss = np.inf
    
    for epoch in trange(1000):
        epoch_losses = []
        for _, batch in enumerate(dataloader):

            # print(scale_factor, tx, ty)
            # print(sf_unnorm, tx_unnorm, ty_unnorm)

            # sf_unnorm = inverse_norm(scale_factor, scale_min, scale_max)
            # tx_unnorm = inverse_norm(tx, tx_min, tx_max)
            # ty_unnorm = inverse_norm(ty, ty_min, ty_max)
            
            # transformation_matrix = get_transformation(sf_unnorm, tx_unnorm, ty_unnorm)
            # print(transformation_matrix)
            # print(transformation_matrix.shape)

            noisy_transformation = gen_noisy_transformations(batch.shape[0], scale_factor, tx, ty)
            # print(noisy_transformation.shape)

            mod_batch = place_patch(batch.to(device), patch.repeat(batch.shape[0], 1, 1, 1), noisy_transformation)
            # print(mod_batch.shape)
            # plt.imsave('misc/dji/temp_batch.jpg', mod_batch[0].permute(1, 2, 0).detach().cpu().numpy())

            results = model(mod_batch)[0]
            boxes_batch = xywh2xyxy(results[:, :, :4])
            scores_batch = results[:, :, 4] * results[:, :, 5]
            soft_scores_batches = torch.nn.functional.softmax(scores_batch * SOFTMAX_SCALE, dim=1)
            
            selected_boxes = torch.bmm(soft_scores_batches.unsqueeze(1), boxes_batch).squeeze(1)
            # print(selected_box.shape)

            # get xyz from bounding box
            xyz = torch.stack([xyz_from_bb(box, camera_intrinsic) for box in selected_boxes])
            # print(xyz.shape)
            # print(xyz[0])

            loss = torch.mean((xyz - target_position.repeat(batch.shape[0], 1))**2)
            epoch_losses.append(loss.detach().cpu().item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            patch.data.clamp_(0, 1.)
            scale_factor.data.clamp_(-1., 1.)
            tx.data.clamp_(-1., 1.)
            ty.data.clamp_(-1., 1.)
        
        if np.mean(epoch_losses) < best_loss:
            best_patch = patch.detach().cpu().numpy()
            best_position = np.hstack([scale_factor.detach().cpu().item(), tx.detach().cpu().item(), ty.detach().cpu().item()])
            #print(best_position, best_position.shape)
            best_loss = np.mean(epoch_losses)


        if epoch % 10 == 0:
            print(f'Epoch {epoch}, loss: {np.mean(epoch_losses)}')
            print(f'sf: {scale_factor}, tx: {tx}, ty: {ty}')
            # place patch in one image and save
            mod_img = place_patch(batch[0].unsqueeze(0).to(device), patch, noisy_transformation[0].unsqueeze(0), random_perspection=False)
            # print(mod_img.shape)
            # draw rectangle of target box
            mod_img = mod_img[0].permute(1, 2, 0).detach().cpu().numpy()
            mod_img = (mod_img * 255).astype(np.uint8)
            mod_img = cv2.cvtColor(mod_img, cv2.COLOR_RGB2BGR)
            # xmin, ymin, xmax, ymax = target_box.detach().cpu().numpy()
            # cv2.rectangle(mod_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 10)
            xmin, ymin, xmax, ymax = selected_boxes[0].detach().cpu().numpy()
            cv2.rectangle(mod_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 3)
            cv2.drawMarker(mod_img, (target_x, target_y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=3)
            
            cv2.imwrite(f'misc/dji/temp_batch_prediction/patch_{epoch}.jpg', mod_img)
        # plt.imsave(f'misc/dji/temp_batch_prediction/patch_{epoch}.jpg', mod_img[0].permute(1, 2, 0).detach().cpu().numpy())
    np.save('misc/dji/optim_patch.npy', best_patch)
    np.save('misc/dji/optim_transformation.npy', best_position)