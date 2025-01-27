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

if __name__ == '__main__':
  
  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load yolov5
    model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True, autoshape=False)
    model.eval()

    scale_factor_height = 320 / 2160
    scale_factor_width = 640 / 3840

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


    scale_factor = torch.tensor([1.]).to(device).requires_grad_(True)
    tx = torch.tensor([60.]).to(device).requires_grad_(True)
    ty = torch.tensor([120.]).to(device).requires_grad_(True)


    opt = torch.optim.Adam([random_patch, scale_factor, tx, ty], lr=3e-2)
    
    for epoch in trange(200):
        epoch_losses = []
        for _, batch in enumerate(dataloader):
            translation_vector = torch.stack([tx, ty]) # shape (2, 1)
            # print(translation_vector.shape)
            eye = torch.eye(2,2).to(device)
            scale = eye * scale_factor  # shape (2,2)

            # print(scale.shape)

            transformation_matrix = torch.cat([scale, translation_vector], dim=1)
            # print(transformation_matrix.shape)

            mod_batch = place_patch(batch.to(device), random_patch.repeat(batch.shape[0], 1, 1, 1), transformation_matrix.repeat(batch.shape[0], 1, 1))
            # print(mod_batch.shape)

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
            scale_factor.data.clamp_(0., 2.3)

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
            
    # batch = next(iter(dataloader))
    # print(batch.shape)

    # transformation_matrix = torch.tensor([[1., 0., 60.], [0., 1., 120.]]).to(device).repeat(batch.shape[0], 1, 1)
    # print(transformation_matrix.shape)

    # mod_batch = place_patch(batch.to(device), random_patch.repeat(batch.shape[0], 1, 1, 1), transformation_matrix, random_perspection=False)
    # for i, mod_img in enumerate(mod_batch):
    #     plt.imsave(f'misc/dji/temp_batch_prediction/mod_{i}.jpg', mod_img.permute(1, 2, 0).detach().cpu().numpy())

    # # predict boxes
    # results = model(batch.to(device))[0]
    # # print(results.shape) 

    # boxes_batch = xywh2xyxy(results[:, :, :4])
    # # print(boxes_batch.shape)
    # scores_batch = results[:, :, 4] * results[:, :, 5]
    # # print(scores_batch.shape)
    # # print(torch.max(scores_batch, dim=1))

    # # scores prediction for each box
    # soft_scores_batches = torch.stack([torch.nn.functional.softmax(score * SOFTMAX_SCALE) for score in scores_batch])
    # # print(soft_scores_batches.shape)

    # # for i, (score, soft_score) in enumerate(zip(scores, soft_scores_batches)):
    # #     axs[i, 0].plot(score.detach().cpu().numpy())
    # #     axs[i, 1].plot(soft_score.detach().cpu().numpy())
    # # plt.savefig('misc/dji/temp_batch_prediction/scores.jpg')



    # selected_box = torch.stack([torch.bmm(soft_score.unsqueeze(0).unsqueeze(0), all_boxes_p_img.unsqueeze(0)) for soft_score, all_boxes_p_img in zip(soft_scores_batches, boxes_batch)]).detach().cpu().squeeze(1).squeeze(1).numpy()
    # #torch.bmm(soft_scores_batches.unsqueeze(1), boxes).squeeze(1).detach().cpu().numpy()
    # print(selected_box)

    # # # max score prediction
    # # max_scores_idx = torch.argmax(scores, dim=1).detach().cpu().numpy()
    # # max_boxes = np.array([boxes[i][idx].detach().cpu().numpy() for i, idx in enumerate(max_scores_idx)])
    # # print(max_boxes.shape)

    # for counter, img in enumerate(batch):
    #     img = img.permute(1, 2, 0).detach().cpu().numpy()
    #     img = (img * 255).astype(np.uint8)
    #     print(img.shape)
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     # scale image to height = 640


    #     # draw box
    #     #scaled_box = scale_box(selected_box[counter], scale_factor_width, scale_factor_height)
    #     xmin, ymin, xmax, ymax = selected_box[counter]
    #     cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 10)
    #     #cv2.rectangle(img, (int(max_boxes[counter][0]), int(max_boxes[counter][1]), int(max_boxes[counter][2]), int(max_boxes[counter][3])), (255, 0, 0), 10)
    #     cv2.imwrite(f'{output_folder}/batch_{counter:04d}.jpg', img)

    # # plot scores

    # for counter, img in enumerate(batch):
    #     img_t = img.unsqueeze(0).to(device)
    #     result = model(img_t)[0]

    #     boxes = xywh2xyxy(result[:, :, :4])
    #     # print(boxes.shape)
    #     scores = result[:, :, 4] * result[0][:, 5]
    #     # print(scores.shape)
    #     print(torch.max(scores), torch.argmax(scores))

    #     soft_scores = torch.nn.functional.softmax(scores * SOFTMAX_SCALE, dim=1)
    #     print(soft_scores.shape)

    #     fig, axs = plt.subplots(2, 2)
    #     axs[0, 0].plot(scores_batch[counter].detach().cpu().numpy())
    #     axs[0, 1].plot(soft_scores_batches[counter].detach().cpu().numpy())
    #     axs[1, 0].plot(scores[0].detach().cpu().numpy())
    #     axs[1, 1].plot(soft_scores[0].detach().cpu().numpy())
    #     plt.savefig(f'misc/dji/temp_batch_prediction/scores_{counter}.jpg')
    #     # difference = torch.mean((soft_scores - soft_scores_batches[counter])**2)
    #     # print(difference)
    #     # print(soft_scores.shape)
    #     selected_box = torch.bmm(soft_scores.unsqueeze(1), boxes).squeeze(1).detach().cpu().numpy()
    #     # print(selected_box.shape)

    #     img = img.permute(1, 2, 0).detach().cpu().numpy()
    #     img = (img * 255).astype(np.uint8)
    #     # print(img.shape)
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     # scale image to height = 640


    #     # draw box
    #     #scaled_box = scale_box(selected_box[counter], scale_factor_width, scale_factor_height)
    #     xmin, ymin, xmax, ymax = selected_box[0]
    #     cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 10)
    #     cv2.imwrite(f'{output_folder}/single_{counter:04d}.jpg', img)

    

    # all_boxes = []

    #for counter, file in enumerate(sorted(glob.glob(f'{images_folder}/*.jpg'))):
    #     img = cv2.imread(file)
    #     cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     # scale image to height = 640
    #     # print(img.shape)
    #     scale_factor_height = 320 / img.shape[0]
    #     scale_factor_width = 640 / img.shape[1]
    #     img_r = cv2.resize(img, (640, 320))

    #     img_t = torch.tensor(img_r).permute(2, 0, 1).unsqueeze(0).float() / 255.
    #     img_t = img_t.requires_grad_(True).to(device)
    #     #print(img_t.shape)
    #     #print(img_t.min(), img_t.max())

    #     # predict boxes
    #     results = model(img_t)[0]  
    #     #print(results.shape)    # batch_size, num_candidates, boxes + class scores 
    #     boxes = xywh2xyxy(results[:, :, :4])
    #     #print(boxes.shape)
    #     scores = results[:, :, 4] * results[0][:, 5] # multiply obj score by person confidence


    #     soft_scores = torch.nn.functional.softmax(scores * SOFTMAX_SCALE)
    #     selected_box = torch.bmm(soft_scores.unsqueeze(1), boxes).squeeze(1).detach().cpu().numpy()
    #     selected_box = scale_box(selected_box[0], scale_factor_width, scale_factor_height).reshape(1, -1)

    #    # print(selected_box)

    #     # draw box
    #     xmin, ymin, xmax, ymax = selected_box[0]
    #     all_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
    #     cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 10)
    #     cv2.imwrite(f'{output_folder}/{counter:04d}.jpg', img)
        
    # print("Saving all boxes...")
    # np.save(f'{output_folder}/all_boxes.npy', np.array(all_boxes))
    # top_scores, top_indices = torch.topk(scores, 5)
    # print(top_scores, top_scores.shape)
    # print(top_indices, top_indices.shape)

    # top_indices = top_indices.squeeze(0)

    # top_box = boxes[0, top_indices[0, 0]].detach().cpu().numpy() # img 0, top candidate
    # print(top_box)

    # select boxes
    # top_boxes = boxes[top_indices[0, 0]].detach().cpu().numpy()
    # print(top_boxes)

    # # draw boxes
    # for idx in top_indices:
    #     top_box = boxes[0, idx].detach().cpu().numpy()
    #     scaled_box = scale_box(top_box, scale_factor_width, scale_factor_height)
    #     xmin, ymin, xmax, ymax = scaled_box
    #     cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 10)

    # cv2.imwrite('misc/dji/output.jpg', img)