import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import yaml
import pickle
from pathlib import Path
import rowan
import torch

import cv2

from util import opencv2quat, load_dataset

RADIUS = 0.5

class Camera:
    def __init__(self, path):
        self.parse(path)
        
    def parse(self, path):
        with open(path) as f:
            camera_config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.camera_intrinsic = np.array(camera_config['camera_matrix'])
        self.distortion_coeffs = np.array(camera_config['dist_coeff'])
        
        rvec = np.array(camera_config['rvec'])
        tvec = camera_config['tvec']

        self.camera_extrinsic = np.zeros((4,4))
        self.camera_extrinsic[:3, :3] = rowan.to_matrix(opencv2quat(rvec))
        self.camera_extrinsic[:3, 3] = tvec
        self.camera_extrinsic[-1, -1] = 1.

        print('extrinsic', np.shape(self.camera_extrinsic))
        print('intrinsic', np.shape(self.camera_intrinsic))
    
    # compute relative position of center of patch in camera frame
    def xyz_from_bb(self, bb):
        # bb - xmin,ymin,xmax,ymax
        # mtrx, dist_vec = get_camera_parameters()
        fx = np.array(self.camera_intrinsic)[0][0]
        fy = np.array(self.camera_intrinsic)[1][1]
        ox = np.array(self.camera_intrinsic)[0][2]
        oy = np.array(self.camera_intrinsic)[1][2]
        # get pixels for bb side center
        P1 = np.array([bb[0],(bb[1] + bb[3])/2])
        P2 = np.array([bb[2],(bb[1] + bb[3])/2])
        # rectify pixels
        P1_rec = cv2.undistortPoints(P1, self.camera_intrinsic, self.distortion_coeffs, None, self.camera_intrinsic).flatten() # distortion is included in my camera intrinsic matrix
        P2_rec = cv2.undistortPoints(P2, self.camera_intrinsic, self.distortion_coeffs, None, self.camera_intrinsic).flatten() # distortion is included in my camera intrinsic matrix

        # get rays for pixels
        a1 = np.array([(P1_rec[0]-ox)/fx, (P1_rec[1]-oy)/fy, 1.0])
        a2 = np.array([(P2_rec[0]-ox)/fx, (P2_rec[1]-oy)/fy, 1.0])
        # normalize rays
        a1_norm = np.linalg.norm(a1)
        a2_norm = np.linalg.norm(a2)
        # get the distance    
        distance = (np.sqrt(2)*RADIUS)/(np.sqrt(1-np.dot(a1,a2)/(a1_norm*a2_norm)))
        # get central ray
        ac = (a1+a2)/2
        # get the position
        xyz = distance*ac/np.linalg.norm(ac)
        return xyz

    def point_from_xyz(self, coords):
        # print('coords shape', np.shape(coords), np.shape(np.array([1])))
        # print(np.shape(self.camera_extrinsic))
        camera_frame = self.camera_extrinsic @ coords
        image_frame = self.camera_intrinsic @ camera_frame[:3]
        u, v, w = image_frame
        img_x = int(np.round(u/w, decimals=0))
        img_y = int(np.round(v/w, decimals=0))
        return (img_x, img_y)


def test_camera():
    dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
    cam_config = 'misc/camera_calibration/calibration.yaml'
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Can be 'yolov5n' - 'yolov5x6', or 'custom'
    cam = Camera(cam_config)

    train_data, train_dataloader = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=False, num_workers=0)

    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    for i in range(10):
        print(i, "===============")
        img = np.array(train_features[i].squeeze())
        label = np.array(train_labels[i])
        # print(np.shape(img))

        results = model(img)
        results = results.pandas().xyxy[0].to_dict(orient="records")
        # print('results', results)
        print('label', label)
        if not np.any(label):
            continue

        img_x, img_y = cam.point_from_xyz(label)
        rgb_img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        cv2.circle(rgb_img, (img_x, img_y), radius=2, color=(0, 255, 0), thickness=1)
        for result in results:
            if result['name'] != 'person':
                continue

            xmin, ymin, xmax, ymax = int(result['xmin']), int(result['ymin']), int(result['xmax']), int(result['ymax'])
            cv2.rectangle(rgb_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            coords = cam.xyz_from_bb((xmin, ymin, xmax, ymax))
            print('coords', coords)
            print('xy', xmin, ymin, xmax, ymax)
            
        
        cv2.imwrite(f'test_cv_{i}.png', rgb_img)


if __name__ == "__main__":
    test_camera()