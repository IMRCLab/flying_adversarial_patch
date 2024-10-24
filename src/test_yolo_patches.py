import yaml
import numpy as np
import cv2
from util import load_dataset
from plots import img_placed_patch
from camera import Camera
import argparse

import torch

PATH = 'results/yolo_patches'
WIDTH = 160
HEIGHT = 96
RESULT_PATH = 'results/yolo_result_imgs'

def load_patches(start, end):
    # patches_on_imgs = []
    # targets = []

    patch_data = {}
    cam_config='misc/camera_calibration/calibration.yaml'
    cam = Camera(cam_config)
    for i in range(start, end+1):
        try:
            with open(f'{PATH}/settings_{i}.yaml') as f:
                settings = yaml.load(f, Loader=yaml.FullLoader)

            num_targets = len(settings['targets']['x'])

            # load all targets in [[x, y, z, 1.],...] format
            targets = [[settings['targets']['x'][i], settings['targets']['y'][0], settings['targets']['z'][i], 1.] for i in range(num_targets)]

            # convert from 3d coords to image coordinates
            target_pxls = [cam.point_from_xyz(t) for t in targets]

            sf, tx, ty = np.load(f'{PATH}/position_norm_{i}.npy')
            patch = np.load(f'{PATH}/last_patch_{i}.npy')

            # place patches on images at correct positions
            imgs_w_patch = 255. * img_placed_patch( targets, 
                                                    patch, 
                                                    scale_norm=sf, 
                                                    tx_norm=tx, 
                                                    ty_norm=ty, 
                                                    p_idx=0, 
                                                    random=True,
                                                    imrc=False)
            

            # save data in dict mapping patch_id to data
            patch_data[i] = (imgs_w_patch, target_pxls)

            print(f'======= Patch {i} ========')
            print('targets', targets)
            print('pixels', target_pxls)
            print('sf', sf)
            print('tx', tx)
            print('ty', ty)
            print("\n")
        except Exception as e:
            print(f'couldn\'t evalulate patch {i}, error: {e}')
    # return patches_on_imgs, targets
    return patch_data
            
def valid_target(pxl):
    return pxl[0] >= 0 and pxl[0] <= WIDTH and pxl[1] >= 0 and pxl[1] <= HEIGHT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start')
    parser.add_argument('--end')
    args = parser.parse_args()

    patch_data = load_patches(int(args.start), int(args.end))

    model = torch.hub.load("ultralytics/yolov5", "yolov5n")

    # BGR
    #          red         blue        green        purple
    colors = [(0, 0, 255),(255, 0, 0),(0, 255, 0),(255, 0, 255)]

    total_loss = 0
    total_valid = 1
    out_of_box = 0
    bad_targets = 0

    # for each patch
    for i in patch_data:
        imgs, targets = patch_data[i][0], patch_data[i][1]

        # for each target for that batch
        for t in range(len(imgs)):
            img, target = imgs[t], targets[t]
            draw_img = img.squeeze()
            draw_img = np.expand_dims(draw_img, -1)
            draw_img = np.repeat(draw_img, 3, -1)
            reshaped_img = np.repeat(img, 3, axis=0)
            results = model(reshaped_img)

            # list: xmin, ymin, xmax, ymax, confidence, class
            # draw every bounding box predicted by yolo (color indicates confidence)
            for j, (xmin, ymin, xmax, ymax, _, _) in enumerate(results.xyxy[0]):

                # check if the target is in the highest confidence bounding box
                if j == 0:
                    # if the target is in the image
                    if valid_target(targets[j]):
                        center_x = (xmin+xmax)//2
                        center_y = (ymin+ymax)//2
                        total_loss += ((center_x - target[0])**2 + (center_y - target[1])**2)**0.5
                        total_valid += 1

                        # if the target is not within the highest confidence bounding box, print and increment counter
                        if target[0] < xmin or target[0] > xmax or target[1] < ymin or target[1] > ymax:
                            print(f'{i} is outside of the yolo box for target {t}')
                            out_of_box += 1
                    else:
                        # invalid target (this shouldn't happen anymore)
                        print(f'BAD TARGET: patch {i}, target {target}')
                        bad_targets += 1

                # draw the bounding box
                cv2.rectangle(draw_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colors[j % len(colors)], 1)
            
            # draw the target
            cv2.circle(draw_img, target, 1, (255, 255, 0), 2)
            cv2.imwrite(f'{RESULT_PATH}/yolo_boxes_{i}_target_{t}.png', draw_img)
        
    print('avg loss', total_loss / total_valid)
    print('total images', total_valid)
    print('out of box', out_of_box)
    print('bad target', bad_targets)
        

