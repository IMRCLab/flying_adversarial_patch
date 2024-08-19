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

from util import opencv2quat, load_dataset, printd

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

        self.fx = np.array(self.camera_intrinsic)[0][0]
        self.fy = np.array(self.camera_intrinsic)[1][1]
        self.ox = np.array(self.camera_intrinsic)[0][2]
        self.oy = np.array(self.camera_intrinsic)[1][2]
    
    def make_extrinsic(self, rvec, tvec):
        self.camera_extrinsic = np.zeros((4,4))
        self.camera_extrinsic[:3, :3] = rowan.to_matrix(opencv2quat(rvec))
        self.camera_extrinsic[:3, 3] = tvec
        self.camera_extrinsic[-1, -1] = 1.

        self.camera_extrinsic_tens = torch.tensor(self.camera_extrinsic, dtype=torch.float32)

    
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
        
        # get pixels for bb side center
        P1 = np.array([bb[0],(bb[1] + bb[3])/2])
        P2 = np.array([bb[2],(bb[1] + bb[3])/2])

        # get rays for pixels
        a1 = np.array([(P1[0]-self.ox)/self.fx, (P1[1]-self.oy)/self.fy, 1.0])
        a2 = np.array([(P2[0]-self.ox)/self.fx, (P2[1]-self.oy)/self.fy, 1.0])

        # normalize rays
        a1_norm = np.linalg.norm(a1)
        a2_norm = np.linalg.norm(a2)
        # print('norms', a1_norm, a2_norm)
        # get the distance    
        distance = (np.sqrt(2)*RADIUS)/(np.sqrt(1-np.dot(a1,a2)/(a1_norm*a2_norm)))
        # print("norm product", a1_norm*a2_norm, np.dot(a1,a2))
        # print('distance', distance)
        # get central ray
        ac = (a1+a2)/2
        # print('ac', ac)
        # get the position
        xyz = distance*ac/np.linalg.norm(ac)
        # print('xyz', xyz)
        # print('test', ac/np.linalg.norm(ac))
        new_xyz = (np.linalg.inv(self.camera_extrinsic) @ [*xyz, 1])[:3]
        return new_xyz

    # compute relative position of center of patch in camera frame
    def tensor_xyz_from_bb(self, bb):
        
        printd('bounding box', bb.grad_fn)
        center = (bb[1] + bb[3])/2

        # get rays for pixels
        a1 = torch.ones(3)
        a1[0] = (bb[0]-self.ox)/self.fx
        a1[1] = (center-self.oy)/self.fy

        a2 = torch.ones(3)
        a2[0] = (bb[2]-self.ox)/self.fx
        a2[1] = (center-self.oy)/self.fy

        # a1 = torch.tensor([(bb[0]-self.ox)/self.fx, (center-self.oy)/self.fy, 1.0])
        # a2 = torch.tensor([(bb[2]-self.ox)/self.fx, (center-self.oy)/self.fy, 1.0])

        printd('a1', a1.grad_fn)

        # normalize rays
        a1_norm = torch.linalg.norm(a1)
        a2_norm = torch.linalg.norm(a2)

        printd('a1 nnorm', a1_norm.grad_fn)

        # get the distance    
        distance = (np.sqrt(2)*RADIUS)/(torch.sqrt(1-torch.dot(a1,a2)/(a1_norm*a2_norm)))

        printd('distance', distance.grad_fn)

        # get central ray
        ac = (a1+a2)/2

        # get the position
        xyz = distance*ac/torch.linalg.norm(ac)

        new_xyz = (torch.linalg.inv(self.camera_extrinsic_tens) @ torch.cat((xyz, torch.ones(1))))[:3]
        return new_xyz
    
    def batch_xyz_from_boxes(self, boxes):
        batch_size = boxes.shape[0]
        xyzs = torch.zeros((batch_size, 3), device=boxes.device)
        # boxes is a tensor of size (B, 4)
        for i in range(batch_size):
            # print('boxes[i]', boxes[i], boxes[i].shape)
            coords = self.tensor_xyz_from_bb(boxes[i])
            printd('coords', coords.grad_fn)
            xyzs[i] = coords

        return xyzs
    
    def batch_xyz_from_bb_bad(self, boxes):
        # boxes (batch_size, 4)
        batch_size = boxes.shape[0]
        centers = (boxes[:, 1] + boxes[:, 3])/2.
        # print('centers', centers.shape)
        # print('ummm', boxes[:, 0].shape)

        a1 = torch.stack(((boxes[:, 0] - self.ox)/self.fx, (centers - self.oy)/self.fy, torch.ones(batch_size)), dim=1)
        a2 = torch.stack(((boxes[:, 2] - self.ox)/self.fx, (centers - self.oy)/self.fy, torch.ones(batch_size)), dim=1)
        print('batch a1', a1)

        a1_norm = torch.linalg.vector_norm(a1, dim=1)
        a2_norm = torch.linalg.vector_norm(a2, dim=1)
        print('batch norms', a1_norm)

        # a1: 4x3, a2: 4x3 and want 4x1
        norm_product = torch.matmul(a1_norm, a2_norm)
        inner_norm = torch.bmm(a1.view(a1.shape[0], 1, a1.shape[1]), a2.view(a1.shape[0], a1.shape[1], 1)).squeeze()
        # .squeeze(-1)
        print('mult norm', norm_product)
        print('inner norm', inner_norm, inner_norm.shape)

        distances = (np.sqrt(2)*RADIUS)/(torch.sqrt(1-inner_norm/norm_product))
        print('distance', distances, distances.shape)

        # # get central ray
        ac = (a1+a2)/2
        ac_norm = torch.linalg.vector_norm(ac, dim=1,keepdim=True)
        print('ac', ac)
        print('ac norm', ac_norm, ac_norm.shape)

        print('test', ac/ac_norm)

        # cam_xyzs = torch.bmm(distances, ac.view(ac.shape[0], 1, ac.shape[1])).squeeze()
        cam_xyzs = distances.view(-1, 1, 1) * (ac/ac_norm)
        print('xyzs', cam_xyzs, cam_xyzs.shape)

        xyzs_homogenous = torch.hstack((cam_xyzs, torch.ones(batch_size, 1)))
        cam_extr_inv = torch.inverse(self.camera_extrinsic_tens)

        xyzs = torch.matmul(xyzs_homogenous, torch.t(cam_extr_inv))
        print('final', xyzs)
        # new_xyz = (np.linalg.inv(self.camera_extrinsic) @ [*xyz, 1])[:3]
        # # /torch.linalg.vector_norm(ac, dim=1)

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
    get_error(cam, dataset_path, model, "calibrated_")

def test_batch_xyz(cam):
    # batch_size = 4
    #                           , 5.3311    -0.18845     0.95316], [2.7442      1.8463      0.2879], [2.8199     -1.2368     0.18378]
    boxes = torch.tensor([[12, 24, 44, 88], [68, 27, 82, 75], [8, 18, 39, 95], [91, 23, 122, 95]])
    # boxes = box.unsqueeze(0)
    # boxes = torch.repeat_interleave(boxes, batch_size, dim=0)

    print('boxes shape', boxes.shape)
    res = cam.batch_xyz_from_bb(boxes)
    print(res)
    
    # print('=======')
    # test_res = cam.xyz_from_bb(box)
    # print(test_res)

    # print('========')
    # # tensor_box = b
    # print(cam.tensor_xyz_from_bb(box))


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

            # print('img shape', np.shape(img))
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
                print(xmin, ymin, xmax, ymax)
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
    # test_camera()
    cam_config = 'misc/camera_calibration/calibration.yaml'
    cam = Camera(cam_config)

    # test_batch_xyz(cam)
    # t = [ 1.0461,  0.3442, -0.3205, 1.]
    t = [ 0.3787, -0.6999,  0.4411, 1.]
    print(cam.point_from_xyz(t))
    # gen_camera_matrix()