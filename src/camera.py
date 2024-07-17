import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import yaml
import pickle
from pathlib import Path
import rowan
import torch

import cv2
import csv

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
        self.make_extrinsic(rvec, tvec)
    
    def make_extrinsic(self, rvec, tvec):
        self.camera_extrinsic = np.zeros((4,4))
        self.camera_extrinsic[:3, :3] = rowan.to_matrix(opencv2quat(rvec))
        self.camera_extrinsic[:3, 3] = tvec
        self.camera_extrinsic[-1, -1] = 1.
    
    def update_with_points(self, objs, imgs, img_size):
        print('updating matrix')
        obj_pts = np.array(objs, dtype=np.float32)[np.newaxis]
        img_pts = np.array(imgs, dtype=np.float32)[np.newaxis]
        cam_matrix = np.zeros((3, 3))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, img_size, cameraMatrix=self.camera_intrinsic, distCoeffs=self.distortion_coeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, img_size, cam_matrix, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        # print(mtx)
        print('new matrix', mtx)
        print('rvecs', rvecs)
        print('tvecs', tvecs)
        self.camera_intrinsic = mtx
        self.distortion_coeffs = dist
        self.make_extrinsic(rvecs[0].squeeze(), tvecs[0].squeeze())

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
        new_xyz = (np.linalg.inv(self.camera_extrinsic) @ [*xyz, 1])[:3]
        return new_xyz

    def point_from_xyz(self, coords):
        camera_frame = self.camera_extrinsic @ coords
        image_frame = self.camera_intrinsic @ camera_frame[:3]
        u, v, w = image_frame
        img_x = int(np.round(u/w, decimals=0))
        img_y = int(np.round(v/w, decimals=0))
        return (img_x, img_y)

def gen_camera_matrix(cam):
    data = csv.DictReader(open('src/camera_calibration/ground_truth_pose.csv', mode='r'))
    obj_pts = []
    img_pts = []
    img_size = (96, 160)
    for d in data:
        obj_pts.append([(d['x']), (d['y']), (d['z'])])
        img_pts.append([d['img_x'], d['img_y']])

    cam.update_with_points(obj_pts, img_pts, img_size)

def test_camera():
    dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Can be 'yolov5n' - 'yolov5x6', or 'custom'
    cam_config = 'misc/camera_calibration/calibration.yaml'
    cam = Camera(cam_config)
    print("======OG CALIBRATION======")
    # get_error(cam, dataset_path, model, "og_")
    # gen_camera_matrix(cam)
    print("========NEW CALIBRATION=====")
    get_error(cam, dataset_path, model, "calibrated_")

def get_error(cam, dataset_path, model, filename):
    
    train_dataloader = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=False, num_workers=0)

    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    print(f"Length: {len(train_dataloader)}")

    total_err = 0
    # for data in train_dataloader:
    for i in range(1):
        train_features, train_labels = next(iter(train_dataloader))

        for i in range(10):
            img = np.array(train_features[i].squeeze())
            label = np.array(train_labels[i])

            results = model(img)
            results = results.pandas().xyxy[0].to_dict(orient="records")
            
            if not np.any(label):
                continue
        
            print(i, "===============")
            print('label', label)

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

                total_err += np.linalg.norm(label[:3] - coords)

            cv2.imwrite(f'{filename}_{i}.png', rgb_img)
    print('total err', total_err)

    # data = csv.DictReader(open('src/camera_calibration/ground_truth_pose.csv', mode='r'))
    # for d in data:
    #     coords = np.array([d['x'], d['y'], d['z'], 0], dtype=np.float32)
    #     calc_x, calc_y = cam.point_from_xyz(coords)
    #     print("==============")
    #     print("calc", calc_x, calc_y)
    #     print(d['img_x'], d['img_y'])


if __name__ == "__main__":
    test_camera()
    # gen_camera_matrix()