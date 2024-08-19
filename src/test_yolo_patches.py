import yaml
import numpy as np
import cv2
from util import load_dataset
from plots import img_placed_patch
from camera import Camera
import argparse

import torch

path = 'results/yolo_patches'
WIDTH = 160
HEIGHT = 96

def load_patches(start, end):
    # patches_on_imgs = []
    # targets = []

    patch_data = {}
    cam_config='misc/camera_calibration/calibration.yaml'
    cam = Camera(cam_config)
    for i in range(start, end+1):
        try:
            with open(f'{path}/settings_{i}.yaml') as f:
                settings = yaml.load(f, Loader=yaml.FullLoader)

            target = [[settings['targets']['x'][0], settings['targets']['y'][0], settings['targets']['z'][0], 1.]]
            target_pxl = cam.point_from_xyz(target[0])

            sf, tx, ty = np.load(f'{path}/position_norm_{i}.npy')
            patch = np.load(f'{path}/last_patch_{i}.npy')

            img_w_patch = 255. * img_placed_patch(  target, 
                                                    patch, 
                                                    scale_norm=sf, 
                                                    tx_norm=tx, 
                                                    ty_norm=ty, 
                                                    p_idx=0, 
                                                    random=True,
                                                    imrc=False)
            img_w_patch = img_w_patch.squeeze(0)

            # patches_on_imgs.append(img_w_patch)
            # targets.append(target_pxl)

            patch_data[i] = (img_w_patch, target_pxl)

            print(f'======= Patch {i} ========')
            print('target', target)
            print('pixel', target_pxl)
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

    # dataset_path = "pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"
    # dataset = load_dataset(dataset_path, batch_size=1, train=False, train_set_size=0.9, IMRC=False)

    patch_data = load_patches(int(args.start), int(args.end))

    model = torch.hub.load("ultralytics/yolov5", "yolov5n")

    # BGR
    #          red         blue 

    ## error: the index i of this is wrong if an exception is thrown when loading patches when a file doesn't exist oops
    colors = [(0, 0, 255),(255, 0, 0),(0, 255, 0),(255, 0, 255)]

    total_loss = 0
    total_valid = 1
    out_of_box = 0
    bad_targets = 0
    for i in patch_data:
        img, target = patch_data[i][0], patch_data[i][1]
        draw_img = img.squeeze()
        draw_img = np.expand_dims(draw_img, -1)
        draw_img = np.repeat(draw_img, 3, -1)
        reshaped_img = np.repeat(img, 3, axis=0)
        results = model(reshaped_img)

        # list: xmin, ymin, xmax, ymax, confidence, class
        for j, (xmin, ymin, xmax, ymax, _, _) in enumerate(results.xyxy[0]):
            if j == 0:
                if valid_target(target):
                    center_x = (xmin+xmax)//2
                    center_y = (ymin+ymax)//2
                    total_loss += ((center_x - target[0])**2 + (center_y - target[1])**2)**0.5
                    total_valid += 1

                    if target[0] < xmin or target[0] > xmax or target[1] < ymin or target[1] > ymax:
                        print(f'{i} is outside of the yolo box')
                        out_of_box += 1
                else:
                    print(f'BAD TARGET: patch {i}, target {target}')
                    bad_targets += 1

            cv2.rectangle(draw_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colors[j % len(colors)], 1)
        
        cv2.circle(draw_img, target, 1, (255, 255, 0), 2)
        cv2.imwrite(f'results/yolo_result_imgs/yolo_boxes_{int(args.start) + i}.png', draw_img)
        
    print(total_loss / total_valid)
    print('out', out_of_box)
    print('bad target', bad_targets)
        

