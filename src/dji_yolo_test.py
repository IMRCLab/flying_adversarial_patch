import torch
import numpy as np
import cv2
import os
import glob

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

    random_patch = torch.rand(1, 3, 80, 80).to(device).requires_grad_(True)
    # print(random_patch.min(), random_patch.max())


    target_box = torch.tensor([480, 125, 619, 241]).to(device)


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


    scale_factor = torch.tensor([0.1]).to(device).requires_grad_(True)
    tx = torch.tensor([-0.7]).to(device).requires_grad_(True)
    ty = torch.tensor([0.3]).to(device).requires_grad_(True)


    opt = torch.optim.Adam([random_patch, scale_factor, tx, ty], lr=3e-2)
    
    for epoch in trange(1000):
        epoch_losses = []
        for _, batch in enumerate(dataloader):
            sf_unnorm = inverse_norm(scale_factor, scale_min, scale_max)
            tx_unnorm = inverse_norm(tx, tx_min, tx_max)
            ty_unnorm = inverse_norm(ty, ty_min, ty_max)




            # print(scale_factor, tx, ty)
            # print(sf_unnorm, tx_unnorm, ty_unnorm)

            translation_vector = torch.stack([tx_unnorm, ty_unnorm]) # shape (2, 1)
            # print(translation_vector.shape)
            eye = torch.eye(2,2).to(device)
            scale = eye * sf_unnorm  # shape (2,2)

            # print(scale.shape)

            transformation_matrix = torch.cat([scale, translation_vector], dim=1)
            # print(transformation_matrix.shape)

            mod_batch = place_patch(batch.to(device), random_patch.repeat(batch.shape[0], 1, 1, 1), transformation_matrix.repeat(batch.shape[0], 1, 1))
            # print(mod_batch.shape)
            #plt.imsave('misc/dji/temp_batch.jpg', mod_batch[0].permute(1, 2, 0).detach().cpu().numpy())

            results = model(mod_batch)[0]
            boxes_batch = xywh2xyxy(results[:, :, :4])
            scores_batch = results[:, :, 4] * results[:, :, 5]
            soft_scores_batches = torch.nn.functional.softmax(scores_batch * SOFTMAX_SCALE, dim=1)
            
            selected_box = torch.bmm(soft_scores_batches.unsqueeze(1), boxes_batch).squeeze(1)
            # print(selected_box.shape)
            
            loss = torch.mean((selected_box - target_box.repeat(batch.shape[0], 1, 1, 1, 1))**2)
            epoch_losses.append(loss.detach().cpu().item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            random_patch.data.clamp_(0, 1)
            scale_factor.data.clamp_(-1., 1.0)
            tx.data.clamp_(-1., 1.)
            ty.data.clamp_(-1., 1.)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, loss: {np.mean(epoch_losses)}')
            print(f'sf: {scale_factor}, tx: {tx}, ty: {ty}')
            # place patch in one image and save
            mod_img = place_patch(batch[0].unsqueeze(0).to(device), random_patch, transformation_matrix.unsqueeze(0), random_perspection=False)
            # print(mod_img.shape)
            # draw rectangle of target box
            mod_img = mod_img[0].permute(1, 2, 0).detach().cpu().numpy()
            mod_img = (mod_img * 255).astype(np.uint8)
            mod_img = cv2.cvtColor(mod_img, cv2.COLOR_RGB2BGR)
            xmin, ymin, xmax, ymax = target_box.detach().cpu().numpy()
            cv2.rectangle(mod_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 10)
            xmin, ymin, xmax, ymax = selected_box[0].detach().cpu().numpy()
            cv2.rectangle(mod_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 10)
            cv2.imwrite(f'misc/dji/temp_batch_prediction/patch_{epoch}.jpg', mod_img)
        # plt.imsave(f'misc/dji/temp_batch_prediction/patch_{epoch}.jpg', mod_img[0].permute(1, 2, 0).detach().cpu().numpy())

    np.save('misc/dji/optim_patch.npy', random_patch.detach().cpu().numpy())
    np.save('misc/dji/optim_transformation.npy', np.array([scale_factor.detach().cpu().numpy(), tx.detach().cpu().numpy(), ty.detach().cpu().numpy()]))