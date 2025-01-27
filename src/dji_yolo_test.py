import torch
import numpy as np
import cv2
import os
import glob

import pickle

import matplotlib.pyplot as plt

from tqdm import tqdm

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


    # create_pkl(images_folder)
    #plt.imsave('misc/dji/temp_load.jpg', all_images[0].transpose(1, 2, 0))


    # load images from pickle
    with open('misc/dji/all_resized_images.pkl', 'rb') as f:
        images = np.array(pickle.load(f))   # shape: (num_images, 3, 320, 640)
    
    print(images.shape)
    print(images[0].min(), images[0].max())
    # # permute to (num_images, 3, 320, 640)
    # images = images.transpose(0, 3, 1, 2)
    # print(images.shape)
    images = torch.tensor(images/255.).float()
    print(images.shape)
    print(images[0].min(), images[0].max())

    # # plot image with matplotlib
    # plt.imsave('misc/dji/temp.jpg', images[0].permute(1, 2, 0).detach().cpu().numpy())

    dataloader = torch.utils.data.DataLoader(images, batch_size=8, shuffle=True)

    batch = next(iter(dataloader))
    print(batch.shape)

    # predict boxes
    results = model(batch.to(device))[0]
    print(results.shape) 

    boxes = xywh2xyxy(results[:, :, :4])
    print(boxes.shape)
    scores = results[:, :, 4] * results[0][:, 5]
    print(scores.shape)

    # scores prediction for each box
    soft_scores_batches = torch.nn.functional.softmax(scores * SOFTMAX_SCALE, dim=1)
    print(soft_scores_batches.shape)
    selected_box = torch.bmm(soft_scores_batches.unsqueeze(1), boxes).squeeze(1).detach().cpu().numpy()
    print(selected_box.shape)

    # # max score prediction
    # max_scores_idx = torch.argmax(scores, dim=1).detach().cpu().numpy()
    # max_boxes = np.array([boxes[i][idx].detach().cpu().numpy() for i, idx in enumerate(max_scores_idx)])
    # print(max_boxes.shape)

    for counter, img in enumerate(batch):
        img = img.permute(1, 2, 0).detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # scale image to height = 640


        # draw box
        #scaled_box = scale_box(selected_box[counter], scale_factor_width, scale_factor_height)
        xmin, ymin, xmax, ymax = selected_box[counter]
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 10)
        #cv2.rectangle(img, (int(max_boxes[counter][0]), int(max_boxes[counter][1]), int(max_boxes[counter][2]), int(max_boxes[counter][3])), (255, 0, 0), 10)
        cv2.imwrite(f'{output_folder}/batch_{counter:04d}.jpg', img)

    
    for counter, img in enumerate(batch):
        img_t = img.unsqueeze(0).to(device)
        result = model(img_t)[0]

        boxes = xywh2xyxy(result[:, :, :4])
        # print(boxes.shape)
        scores = result[:, :, 4] * result[0][:, 5]
        # print(scores.shape)

        soft_scores = torch.nn.functional.softmax(scores * SOFTMAX_SCALE, dim=1)
        difference = torch.mean((soft_scores - soft_scores_batches[counter])**2)
        print(difference)
        # print(soft_scores.shape)
        selected_box = torch.bmm(soft_scores.unsqueeze(1), boxes).squeeze(1).detach().cpu().numpy()
        # print(selected_box.shape)

        img = img.permute(1, 2, 0).detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        # print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # scale image to height = 640


        # draw box
        #scaled_box = scale_box(selected_box[counter], scale_factor_width, scale_factor_height)
        xmin, ymin, xmax, ymax = selected_box[0]
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 10)
        cv2.imwrite(f'{output_folder}/single_{counter:04d}.jpg', img)



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