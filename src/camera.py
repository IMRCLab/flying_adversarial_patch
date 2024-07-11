import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import yaml
import pickle
from pathlib import Path
import rowan

import cv2

from util import opencv2quat, load_dataset

RADIUS = 0.1405

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

def test_camera():
    dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'

    train_dataloader = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=False, num_workers=0)

    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img)
    plt.savefig('test.png')
    print(label)

if __name__ == "__main__":
    test_camera()