import numpy as np
import argparse

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import yaml
import rowan

from pathlib import Path

import torch
import cv2

from util import opencv2quat
from patch_placement import _perspective_grid

def gen_transformation_matrix(sf, tx, ty):
    matrix = np.zeros((3,3))
    matrix[0,0] = sf
    matrix[1,1] = sf
    matrix[0, 2] = tx
    matrix[1, 2] = ty
    matrix[2, 2] = 1.
    return matrix

# rotation vectors are axis-angle format in "compact form", where
# theta = norm(rvec) and axis = rvec / theta
# they can be converted to a matrix using cv2. Rodrigues, see
# https://docs.opencv.org/4.7.0/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
# def opencv2quat(rvec):
#     angle = np.linalg.norm(rvec)
#     if angle == 0:
#         q = np.array([1,0,0,0])
#     else:
#         axis = rvec.flatten() / angle
#         q = rowan.from_axis_angle(axis, angle)
#     return q

def get_bb_patch(transformation):
    transformation_t = torch.tensor(transformation, dtype=torch.double).unsqueeze(0)
    print(transformation_t, transformation_t.shape)
    inv_t_matrix = torch.inverse(transformation_t)
    # print("inverted matrix shape: ", inv_t_matrix.shape)
    batch_coeffs = inv_t_matrix.reshape(inv_t_matrix.shape[0], -1) # flatten matrices
    # print("coeffs shape: ", batch_coeffs.shape)
    device = torch.device('cpu')
    batch_grid = _perspective_grid(batch_coeffs, w=80, h=80, ow=160, oh=96, dtype=torch.double, device=device, center = [1., 1.])
    print(batch_grid, batch_grid.shape)

    # inv_transform = torch.inverse(transformation_t)[:, :2]
    # affine_grid = torch.nn.functional.affine_grid(inv_transform, size=(1, 1, 96, 160), align_corners=True)

    transformed_patch = torch.nn.functional.grid_sample(torch.ones(1, 1, 96, 160).double(), batch_grid, align_corners=True, padding_mode='zeros')

    patch_coords = torch.nonzero(transformed_patch)[..., 2:]
    # print(patch_coords, patch_coords.shape)

    xmin = patch_coords[0][1].item()
    ymin = patch_coords[0][0].item()
    xmax = patch_coords[-1][1].item()
    ymax = patch_coords[-1][0].item()


    
    new_width = (xmax - xmin)
    new_height = (ymax - ymin)
    original_aspect_ratio = 160 / 96

    if xmin == 0.:
        if ymin > 0. and ymax<95.:
            xmin = xmax - (original_aspect_ratio * new_height)
        else:
            print("Can't calculate bounding box properly!")
            return
            # raise()

    # fig, ax = plt.subplots()
    # ax.imshow(transformed_patch[0][0], cmap='gray')
    # plt.show()

    return [xmin, ymin, xmax, ymax]

# printed patch width = 28.2 cm
RADIUS = 0.1405 # in m

# compute relative position of center of patch in camera frame
def xyz_from_bb(bb,mtrx, dist_coeffs):
    # bb - xmin,ymin,xmax,ymax
    # mtrx, dist_vec = get_camera_parameters()
    fx = np.array(mtrx)[0][0]
    fy = np.array(mtrx)[1][1]
    ox = np.array(mtrx)[0][2]
    oy = np.array(mtrx)[1][2]
    # get pixels for bb side center
    P1 = np.array([bb[0],(bb[1] + bb[3])/2])
    P2 = np.array([bb[2],(bb[1] + bb[3])/2])
    # rectify pixels
    P1_rec = cv2.undistortPoints(P1, mtrx, dist_coeffs, None, mtrx).flatten() # distortion is included in my camera intrinsic matrix
    P2_rec = cv2.undistortPoints(P2, mtrx, dist_coeffs, None, mtrx).flatten() # distortion is included in my camera intrinsic matrix

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



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path')


    args = parser.parse_args()

    patches = np.load(Path(args.path) / 'patches.npy')[-1]
    M = patches.shape[0]

    # # print(patches.shape)

    with PdfPages(Path(args.path) / 'print.pdf') as pdf:
        for patch_idx, patch in enumerate(patches):
            fig, ax = plt.subplots(figsize=(11.69,8.27))
            ax.imshow(patch[0], cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    # If your printer also only handles jpgs, use this code instead.
    # fig, ax = plt.subplots(figsize=(11.69,8.27))
    # ax.imshow(patch, cmap='gray')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig(Path(args.path) / 'print_patch.jpg', dpi=100)


    dict = {'targets': {'x': [], 'y':[], 'z':[]}, 'assignment_patch': [], 'patch_in_victim': {'x': [], 'y':[], 'z':{}}, 'patch_pos': {'sf':[], 'tx':[], 'ty':[]}}

    with open(Path(args.path) / 'settings.yaml') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    dict['targets'] = settings['targets']
    K = (len(list(settings['targets'].values())[0]))

    positions = np.load(Path(args.path) / 'positions_norm.npy')[:, -1]
    # print(positions.shape)
    positions = np.swapaxes(positions, 1, 2)
    positions = np.squeeze(positions, axis=3)
    # positions = np.reshape(positions, (M, K, 3))

    # print(positions)
    # print(positions.shape)

    A = np.load(Path(args.path) / 'stats_p.npy')[-1]

    # print(A)
    A = np.where(A >= 0.5, True, False)
    # print(A)

    # print(A[..., 1])

    # get patch indices according to assignment
    K_target_patch = [np.argwhere(A[..., i] == True).item() for i in range(A.shape[1])]
    print(K_target_patch)
    dict['assignment_patch'] = K_target_patch

    K_target_position = np.zeros((K, 3))
    for k in range(K):
        K_target_position[k, 0] = positions[0][..., k][K_target_patch[k]]
        K_target_position[k, 1] = positions[1][..., k][K_target_patch[k]]
        K_target_position[k, 2] = positions[2][..., k][K_target_patch[k]]

    # print(K_target_position[0])
    dict['patch_pos']['sf'] = K_target_position[..., 0].tolist()
    dict['patch_pos']['tx'] = K_target_position[..., 1].tolist()
    dict['patch_pos']['ty'] = K_target_position[..., 2].tolist()

    # dict['patch_pos']['sf'] = positions[0].tolist()#positions[:, :, 0, 0].tolist()
    # dict['patch_pos']['tx'] = positions[1].tolist()#positions[:, :, 1, 0].tolist()
    # dict['patch_pos']['ty'] = positions[2].tolist()#positions[:, :, 2, 0].tolist()

    # print(dict)

    with open('misc/camera_calibration/calibration.yaml') as f:
        camera_config = yaml.load(f, Loader=yaml.FullLoader)

    camera_intrinsic = np.array(camera_config['camera_matrix'])
    distortion_coeffs = np.array(camera_config['dist_coeff'])
    
    rvec = np.array(camera_config['rvec'])
    tvec = camera_config['tvec']

    camera_extrinsic = np.zeros((4,4))
    camera_extrinsic[:3, :3] = rowan.to_matrix(opencv2quat(rvec))
    camera_extrinsic[:3, 3] = tvec
    camera_extrinsic[-1, -1] = 1.

    
    T_patch_victim = np.zeros((K, 3))

    for k in range(K):
        transformation_matrix = gen_transformation_matrix(*K_target_position[k])
        print(transformation_matrix, transformation_matrix.shape)
        bounding_box_placed_patch = get_bb_patch(transformation_matrix)
        patch_in_camera = xyz_from_bb(bounding_box_placed_patch, camera_intrinsic, distortion_coeffs)
        patch_in_victim = (np.linalg.inv(camera_extrinsic) @ [*patch_in_camera, 1])[:3]
        T_patch_victim[k, :] = patch_in_victim
    
    

    dict['patch_in_victim']['x'] = T_patch_victim.T[0].tolist()
    dict['patch_in_victim']['y'] = T_patch_victim.T[1].tolist()
    dict['patch_in_victim']['z'] = T_patch_victim.T[2].tolist()

    # print(dict)

    with open(Path(args.path) / 'results.yaml', 'w') as file:
        yaml.dump(dict, file)

    with open('results.yaml', 'w') as file:
        yaml.dump(dict, file)