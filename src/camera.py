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

IM_HEIGHT = 96
IM_WIDTH = 160
person_width = 10


"""
Radius: 0.22650666534900665, Softmax Mult: 0.9287999868392944
Distance YOLO to Frontnet Mean:  0.44556783571005215
Original fx, fy, ox, oy:  0.7463185099939119 0.7129203609802661 0.6851143101833982 0.5121119939461743
Current fx, fy, ox, oy:  0.8410791925161974 0.7131281085250916 0.7668779826941132 0.5191798133367411
===================================
Original extrinsic:  tensor([[ 5.23360e-02, -9.98630e-01, -1.11022e-16,  0.00000e+00],
        [ 1.73410e-01,  9.08804e-03, -9.84808e-01,  0.00000e+00],
        [ 9.83458e-01,  5.15409e-02,  1.73648e-01, -2.50000e-02],
        [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]], device='cuda:0', dtype=torch.float64)
Current extrinsic:  tensor([[ 6.18377e-02, -1.02535e+00, -4.59242e-03, -8.95689e-05],
        [ 1.78357e-01,  3.03294e-03, -9.79328e-01,  3.99726e-03],
        [ 9.55689e-01, -9.49713e-03,  1.83065e-01, -2.08924e-01],
        [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]], device='cuda:0', dtype=torch.float64, grad_fn=<StackBackward0>)

"""

class Camera:
    def __init__(self, path, device='cpu'):
        self.device = device
        self.fx = torch.tensor(84.1079, device=self.device, dtype=torch.float32)
        self.fy = torch.tensor(71.3128, device=self.device, dtype=torch.float32)
        self.ox = torch.tensor(76.6878, device=self.device, dtype=torch.float32)
        self.oy = torch.tensor(51.9179, device=self.device, dtype=torch.float32)
        self.radius = torch.tensor(0.22650666534900665, device=self.device, dtype=torch.float32)  # meters
        self.camera_intrinsic = np.array([[self.fx.item(), 0, self.ox.item()],
                                            [0, self.fy.item(), self.oy.item()],
                                            [0, 0, 1]], dtype=np.float32)
        self.camera_intrinsic_tens = torch.tensor(self.camera_intrinsic, dtype=torch.float32, device=device)


        self.camera_extrinsic = np.array([[ 6.18377e-02, -1.02535e+00, -4.59242e-03, -8.95689e-05],
                                            [ 1.78357e-01,  3.03294e-03, -9.79328e-01,  3.99726e-03],
                                            [ 9.55689e-01, -9.49713e-03,  1.83065e-01, -2.08924e-01],
                                            [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]], dtype=np.float32)
        self.camera_extrinsic_tens = torch.tensor(self.camera_extrinsic, dtype=torch.float32, device=self.device)




        # with open(path) as f:
        #     camera_config = yaml.load(f, Loader=yaml.FullLoader)

        # self.camera_intrinsic = np.array(camera_config['camera_matrix'])
        # # self.distortion_coeffs = np.array(camera_config['distortion_coeffs'])
        # self.camera_intrinsic_tens = torch.tensor(self.camera_intrinsic, dtype=torch.float32, device=device)
        
        # self.device = device
        # # print("Camera device:", self.device)
        
        # rvec = np.array(camera_config['rvec'])
        # tvec = camera_config['tvec']
        # self.make_extrinsic(rvec, tvec)

        # self.fx = torch.tensor(self.camera_intrinsic[0][0], device=self.device, dtype=torch.float32)
        # self.fy = torch.tensor(self.camera_intrinsic[1][1], device=self.device, dtype=torch.float32)
        # self.ox = torch.tensor(self.camera_intrinsic[0][2], device=self.device, dtype=torch.float32)
        # self.oy = torch.tensor(self.camera_intrinsic[1][2], device=self.device, dtype=torch.float32)

        

        # self.radius = torch.tensor(0.2868165075778961, device=self.device, dtype=torch.float32)  # meters

    def make_extrinsic(self, rvec, tvec):
        self.camera_extrinsic = np.zeros((4,4))
        self.camera_extrinsic[:3, :3] = rowan.to_matrix(opencv2quat(rvec))
        self.camera_extrinsic[:3, 3] = tvec
        self.camera_extrinsic[-1, -1] = 1.

        self.camera_extrinsic_tens = torch.tensor(self.camera_extrinsic, dtype=torch.float32, device=self.device)

        

    # originally for updating the calibration using new ground truth data, not used for anything right now
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
        
        # get pixels for bb side center
        P1 = np.array([bb[0],(bb[1] + bb[3])/2])
        P2 = np.array([bb[2],(bb[1] + bb[3])/2])

        # print(P1, P2)

        # get rays for pixels
        a1 = np.array([(P1[0]-self.ox)/self.fx, (P1[1]-self.oy)/self.fy, 1.0])
        a2 = np.array([(P2[0]-self.ox)/self.fx, (P2[1]-self.oy)/self.fy, 1.0])

        # normalize rays
        a1_norm = np.linalg.norm(a1)
        a2_norm = np.linalg.norm(a2)

        # get the distance    
        distance = (np.sqrt(2)*0.3)/(np.sqrt(1-np.dot(a1,a2)/(a1_norm*a2_norm)))

        ac = (a1+a2)/2

        # get the position
        xyz = distance*ac/np.linalg.norm(ac)
        new_xyz = (np.linalg.inv(self.camera_extrinsic) @ [*xyz, 1])[:3]
        return new_xyz

    # compute relative position of center of patch in camera frame
    def tensor_xyz_from_bb(self, bb):
        # center row
        # print("bb:", bb.device, bb.dtype)
        center = (bb[1] + bb[3]) / 2

        # build rays without in-place ops to keep autograd graph
        a1x = (bb[0] - self.ox) / self.fx
        a1y = (center - self.oy) / self.fy
        a2x = (bb[2] - self.ox) / self.fx
        a2y = (center - self.oy) / self.fy

        one = torch.ones_like(a1x)
        a1 = torch.stack((a1x, a1y, one))
        a2 = torch.stack((a2x, a2y, one))

        # normalize rays
        a1_norm = torch.linalg.norm(a1)
        a2_norm = torch.linalg.norm(a2)

        # distance on the circle of radius (differentiable)
        sqrt2 = torch.sqrt(torch.tensor(2.0, device=bb.device, dtype=bb.dtype))

        cosang = torch.dot(a1, a2) / (a1_norm * a2_norm + 1e-12)
        distance = sqrt2 * self.radius / torch.sqrt(1.0 - cosang + 1e-12)

        # central ray and xyz
        ac = (a1 + a2) * 0.5
        xyz = distance * ac / (torch.linalg.norm(ac) + 1e-12)

        # transform to world (constants are tensors, op remains differentiable wrt xyz)
        # print("xyz:", xyz.device, xyz.dtype)
        # print("camera extrinsic:", self.camera_extrinsic_tens.device, self.camera_extrinsic_tens.dtype)
        new_xyz = (torch.linalg.inv(self.camera_extrinsic_tens) @ torch.cat((xyz, torch.ones(1, device=xyz.device, dtype=xyz.dtype))))[:3]
        return new_xyz
    
    def batch_xyz_from_boxes(self, boxes):
        # build per-sample outputs and stack to preserve graph
        outs = []
        for i in range(boxes.shape[0]):
            coords = self.tensor_xyz_from_bb(boxes[i])
            yaw = torch.atan2(coords[1], coords[0])
            outs.append(torch.cat((coords, yaw.unsqueeze(0))))
        return torch.stack(outs, dim=0)

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


# test stuff
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
    # invalid = 0
    # total = 0
    # for x in np.linspace(0, 2, 100):
    #     for y in np.linspace(-1, 1, 100):
    #         for z in np.linspace(-0.5, 0.5, 50):
    #             # print(x)
    #             total += 1
    #             t = [ x, y, z, 1.]
    #             img_x, img_y = cam.point_from_xyz(t)
    #             # print(x, y)
    #             if img_x < 0 or img_x > 196 or img_y < 0 or img_y > 96:
    #                 invalid += 1
    #                 print(t)
    # print('invalid', invalid)
    # print('total', total)
    # print(invalid/total)

    invalid = 0
    total = 0
    for img_x in range(10, 160):
        for img_y in range(15, 96):
            # print(x)
            total += 1

            x, y, z = cam.xyz_from_bb((img_x - 10, img_y - 15, img_x + 10, img_y + 15))

            if x < 0 or x > 2 or y < -1 or y > 1 or z < -0.5 or z > 0.5:
                print(x, y, z)
                invalid += 1

    print('invalid', invalid)
    print('total', total)
    print(invalid/total)

    # s = 4
    # pxl = (1, 1, 3, 3)
    # xyz = cam.xyz_from_bb(pxl)
    # print(xyz)
    # gen_camera_matrix()