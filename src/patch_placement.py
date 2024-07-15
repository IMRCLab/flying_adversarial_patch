import numpy as np
import torch
#from rowan import to_matrix
from torch.nn.functional import grid_sample

from torchvision.transforms import RandomPerspective
from torchvision.transforms.v2.functional._geometry import _apply_grid_transform
from typing import List


def to_rotation_matrix(q, require_unit=True):
    # copy of rowan's to_matrix() function and adapting it for torch
    # source: https://github.com/glotzerlab/rowan/blob/1b64ac7399e86459ee95e8499b11919b83a30305/rowan/functions.py#L952
    s = torch.linalg.norm(q)
    if torch.any(s == 0.0):
        raise ZeroDivisionError("At least one element of q has approximately zero norm")
    elif require_unit and not torch.allclose(s, torch.tensor(1.0)):
        raise ValueError(
            "Not all quaternions in q are unit quaternions. \
If this was intentional, please set require_unit to False when \
calling this function."
        )

    m = torch.empty(q.shape[:-1] + (3, 3))
    s = torch.pow(s, -1.0)  # For consistency with Wikipedia notation
    m[..., 0, 0] = 1.0 - 2 * s * (q[..., 2] ** 2 + q[..., 3] ** 2)
    m[..., 0, 1] = 2 * (q[..., 1] * q[..., 2] - q[..., 3] * q[..., 0])
    m[..., 0, 2] = 2 * (q[..., 1] * q[..., 3] + q[..., 2] * q[..., 0])
    m[..., 1, 0] = 2 * (q[..., 1] * q[..., 2] + q[..., 3] * q[..., 0])
    m[..., 1, 1] = 1.0 - 2 * (q[..., 1] ** 2 + q[..., 3] ** 2)
    m[..., 1, 2] = 2 * (q[..., 2] * q[..., 3] - q[..., 1] * q[..., 0])
    m[..., 2, 0] = 2 * (q[..., 1] * q[..., 3] - q[..., 2] * q[..., 0])
    m[..., 2, 1] = 2 * (q[..., 2] * q[..., 3] + q[..., 1] * q[..., 0])
    m[..., 2, 2] = 1.0 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2)
    return m


def calc_T_in_attaker_frame(patch_size, scale_factor=0.01, xyz_translations=[0., 0., 0.]):
    """
    Calculate the matrix for translating the patch into the attacker UAV frame.
    Parameters:
        ----------
        patch_size: list, [height, width] of the patch
        scale_factor: float, set the scaling factor of the patch 
        xyz_translation: list, add a translation in [x, y, z]-direction
    Returns:
        a (4,4) numpy array, the transformation matrix
    """
    # the center of the patch will first be set to the center of the attacker
    # therefore, the upper left corner is shifted from (0,0) to (-height/2, -width/2)
    t_x = - patch_size[0] / 2
    t_y = - patch_size[1] / 2
    # no translation in z direction needed currently
    t_z = 0

    # adjust translation with scaling factor
    t_x *= scale_factor
    t_y *= scale_factor
    t_z *= scale_factor

    # now add the translations in x, y and z direction according 
    # to the exact placement of the patch on the attacker
    t_x = t_x + xyz_translations[0]
    t_y = t_y + xyz_translations[1]
    t_z = t_z + xyz_translations[2]

    # TODO: add calculation of rotation
    # for now, no rotation is performed


    T_patch_in_attacker = torch.tensor([[scale_factor*1., 0., 0., t_x],
                                    [0., scale_factor*1., 0., t_y],
                                    [0., 0., scale_factor*1., t_z],
                                    [0., 0., 0., 1.]])
    
    return T_patch_in_attacker

def calc_T_attacker_in_camera(attacker_xyz, attacker_quaternions):
    """
    Calulate the matrix translating the 3D coordinates of the attacker UAV
    into camera frame given the quaternions.
    Parameters:
        ----------
        attacker_xyz: a (3,) numpy array, the 3D coordinates of the attacker UAV
        attacker_quaternions: a (4,) numpy array, the quaternions (qx, qy, qz, qw) 
                              describing the rotation of the attacker UAV to the camera
    Returns:
        a (4,4) numpy array, the transformation matrix
    """
    T_attacker_in_camera = torch.zeros((4,4))
    # calculate rotation matrix from quaternions with rowan's to_matrix
    rotation = to_rotation_matrix(attacker_quaternions)
    # fill empty matrix with values for rotation and translation 
    T_attacker_in_camera[:3, :3] = rotation
    T_attacker_in_camera[:, 3][:3] = attacker_xyz
    T_attacker_in_camera[3, 3] = 1.
    
    return T_attacker_in_camera

def _perspective_grid(
    coeffs: List[float], 
    w: int, h: int, 
    ow: int, oh: int, 
    dtype: torch.dtype, 
    device: torch.device,
    center = None,
) -> torch.Tensor:
    # source: https://github.com/pytorch/pytorch/issues/100526#issuecomment-1610226058
    # https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/
    # src/libImaging/Geometry.c#L394

    #
    # x_out = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / (coeffs[6] * x + coeffs[7] * y + 1)
    # y_out = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / (coeffs[6] * x + coeffs[7] * y + 1)
    #
    batch_size = coeffs.shape[0]
    theta1 = coeffs[..., :6].reshape(batch_size, 2, 3)

    theta2 = coeffs[..., 6:].repeat_interleave(2, dim=0) # theta2 is a matrix of shape [2, 3], it is the last row of the original transformation matrix repeated 2x
    theta2 = theta2.reshape(batch_size, 2, 3) # reshape from [batch_size*2, 3] to [batch_size, 2, 3] 

    d = 0.5
    base_grid = torch.empty(batch_size, oh, ow, 3, dtype=dtype, device=device)
    x_grid = torch.linspace(d, ow + d - 1.0, steps=ow, device=device, dtype=dtype)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(d, oh + d - 1.0, steps=oh, device=device, dtype=dtype).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta1 = theta1.transpose(1, 2).div_(torch.tensor([0.5 * w, 0.5 * h], dtype=dtype, device=device))
    shape = (batch_size, oh * ow, 3)
    output_grid1 = base_grid.view(shape).bmm(rescaled_theta1)
    output_grid2 = base_grid.view(shape).bmm(theta2.transpose(1, 2))

    if center is not None:
        center = torch.tensor(center, dtype=dtype, device=device)
    else:
        center = 1.0

    output_grid = output_grid1.div_(output_grid2).sub_(center)
    return output_grid.view(batch_size, oh, ow, 2)
    
def get_bit_mask(patch_size, image_size, grid):
    """"
    Calculate a bit mask for replacing the pixels of the transformed patch in the original image.
    Parameters:
        ----------
        image_coords: list of numpy arrays [img_x, img_y], the projected 2D pixel coordinates of the patch
        image_size: list, [height, width] of the original image
    Returns:
        a (height, width) numpy array, the bit mask for placing the patch
    """
    bit_mask = torch.ones((image_size[2:]))

    #ori_x = torch.linspace(0, patch_size[2])
    #ori_y = torch.linspace(0, patch_size[3])


    # for x in range(image_size[2]):
    #     for y in range(image_size[3]):
    #         if x,y in grid:
    #             bit_mask[x,y] = patch[ori_x[x], ori_y[y]]

    # for k in range(patch_size[2]):
    #     for j in range(patch_size[3]):
    #         print(max(0, 1-torch.abs(grid[0]-j)))
    #         print(max(0, 1-torch.abs(grid[1]-k)))
    


    # for x in range(image_size[2]):
    #     for y in range(image_size[3]):
    #         bit_mask[x,y] = 
    # for x in range(image_size[2]):
    #     for y in range(image_size[3]):


    #img_x, img_y = grid.reshape(2, 100, 100)

    # for x in range(img_x.shape[0]):
    #     for y in range(img_y.shape[0]):
    #         # only replace pixels that are actually visible in the image
    #         # this means we only replace pixels starting from the upper left corner (0,0)
    #         # until the lower right corner (height, width) of the original image
    #         # any pixels outside of the original image are ignored
    #         if img_x[x][y] >= 0. and img_y[x][y] >= 0.: 
    #             if img_x[x][y] < image_size[2] and img_y[x][y] < image_size[3]:
    #                 # print(x, y)
    #                 # print(img_x[x,y], img_y[x,y])
    #                 bit_mask[int(img_x[x][y])][int(img_y[x][y])] = 0.


    #img = grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)
    # get ones where there's a patch in the image
    #ones = torch.ones(patch_size)
    #bit_mask = grid_sample(ones, grid, mode="nearest", padding_mode="zeros", align_corners=False)

    # invert the bit mask, such that there's zeros where there's a patch and ones where it isn't
    #bit_mask = 1 - bit_mask

    #plt.imshow(bit_mask.detach().numpy())
    return bit_mask

def get_transformed_patch(grid, patch, image_size):
    """"
    Place all pixel values of the patch at the correct calculated position in a black image
    (for easier addition to the original image).
    Parameters:
        ----------
        image_coords: list of numpy arrays [img_x, img_y], the projected 2D pixel coordinates of the patch
        patch: a numpy array, the current iteration of the patch
        image_size: list, [height, width] of the original image
    Returns:
        a (height, width) numpy array, the placed patch in an otherwise black image
    """
    transformed_patch = torch.zeros((image_size[2:]))

    #transformed_patch = grid_sample(patch, grid, mode="nearest", padding_mode="zeros", align_corners=False)


    # for ind_x, x in enumerate(img_x):
    #     for ind_y, y in enumerate(img_y):
    #         if ind_x
    #         image[][] = patch[ind_x][ind_y]
    # print("--beginning of transformed patch---")
    # print(grid.shape)
    # print(grid.reshape(2, 100, 100).shape)
    img_x, img_y = grid.reshape(2, 100, 100)


    for x in range(img_x.shape[0]):
        for y in range(img_y.shape[1]):
            # only replace pixels that are actually visible in the image
            # this means we only replace pixels starting from the upper left corner (0,0)
            # until the lower right corner (height, width) of the original image
            # any pixels outside img_x, img_y
            transformed_patch[img_x[x][y]][img_y[x][y]] = patch[x][y]

    return transformed_patch
    

def place_patch(image, patch, transformation_matrix, random_perspection=True):
    """"
    Place all pixel values of the patch at the correct calculated position in the original image.
    Parameters:
        ----------
        image: a numpy array, the original image
        patch: a numpy array, the current iteration of the patch
        attacker_pose: a (7,) numpy array, including the 3D coordinates of the attacker UAV and the quaternions
        camera_config: dict, includes the camera intrinsics, translation matrix and distortion coefficients
    Returns:
        a numpy array, the final manipulated image including the placed patch
    """
    # print("--inside place patch--")
    p_height, p_width = patch.shape[-2:]
    # print("patch shape: ", p_height, p_width)
    i_height, i_width = image.shape[-2:]

    mask = torch.ones_like(patch)

    # PyTorch's affine grid funtion needs the inverse of the 3x3 transformation matrix
    # transformation_matrix = torch.cat((transformation_matrix, torch.tensor([[[0, 0, 1]]], device=transformation_matrix.device)), dim=1)
    # last_row = torch.tensor([[0, 0, 1]], device=transformation_matrix.device)
    # transformation_matrix = torch.stack([torch.cat([transformation_matrix[i], last_row]) for i in range(len(transformation_matrix))])
    # inv_t_matrix = torch.inverse(transformation_matrix)[:, :2] # affine grid expects only the first 2 rows, the last row (0, 0, 1) is neglected
    # affine_grid = torch.nn.functional.affine_grid(inv_t_matrix, size=(len(transformation_matrix), 1, 96, 160), align_corners=False)

    # # calculate both the bit mask and the transformed patch
    # bit_mask = grid_sample(mask, affine_grid, mode='bilinear', align_corners=False, padding_mode='zeros').bool()
    # transformed_patch = grid_sample(patch, affine_grid, mode='bilinear', align_corners=False, padding_mode='zeros')

    # new perspective grid implementation
    # can only transform single image now!
    last_row = torch.tensor([[0, 0, 1]], device=transformation_matrix.device)
    transformation_matrix = torch.stack([torch.cat([transformation_matrix[i], last_row]) for i in range(len(transformation_matrix))])
    inv_t_matrix = torch.inverse(transformation_matrix)
    # print("inverted matrix shape: ", inv_t_matrix.shape)
    batch_coeffs = inv_t_matrix.reshape(inv_t_matrix.shape[0], -1) # flatten matrices
    # print("coeffs shape: ", batch_coeffs.shape)
    batch_grid = _perspective_grid(batch_coeffs, w=p_width, h=p_height, ow=i_width, oh=i_height, dtype=torch.float32, device=patch.device, center = [1., 1.])

    # bit_mask = torch.stack([_apply_grid_transform(m, grid, mode="nearest", fill=0) for m, grid in zip(mask, batch_grid)])
    # transformed_patch = torch.stack([_apply_grid_transform(p, grid, mode="nearest", fill=0) for p, grid in zip(patch, batch_grid)])
    bit_mask = grid_sample(mask, batch_grid, mode='bilinear', align_corners=False, padding_mode='zeros').bool()
    transformed_patch = grid_sample(patch, batch_grid, mode='bilinear', align_corners=False, padding_mode='zeros')

    # print("masks shape: ", bit_mask.shape)
    # print("patches shape: ", transformed_patch.shape)

    if random_perspection:
        random_rotations = RandomPerspective(distortion_scale=0.2, p=0.9)
        perspected_both = random_rotations(torch.cat([transformed_patch, bit_mask])) # apply same transformation to patch and bit mask
        perspected_patch, perspected_mask = torch.split(perspected_both, len(transformation_matrix)) # split into patch batch and mask batch

        # first erase all pixel values in the original image in the area of the patch
        modified_image = image * ~perspected_mask.bool()
        # transformed_patch *= bit_mask
        # and now replace these values with the transformed patch
        modified_image += perspected_patch
    else:
        # first erase all pixel values in the original image in the area of the patch
        modified_image = image * ~bit_mask.bool()
        # and now replace these values with the transformed patch
        modified_image += transformed_patch

    return modified_image

if __name__=="__main__":
    # ---Example for patch placement---
    import matplotlib.pyplot as plt
    # load the Frontnet dataset
    from util import load_dataset
    dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
    dataset = load_dataset(path=dataset_path, batch_size=32, shuffle=False, drop_last=True, num_workers=0)

    # choose an image from the dataset
    # here, we chose the first pair (image, pose)
    image, pose = dataset.dataset.__getitem__(0)

    # the image is in shape (color channels, height, width)
    # since the images are grayscale, the color channel == 1
    # we'll extend the shape by one dimension to work with batches of images -> there's only one image so batch_no == 1
    image = image.unsqueeze(0)
    # the pixel values of the images range from 0 to 255
    # pose is stored as x, y, z, phi -> phi is not needed for patch placement
    pose = pose[:3]

    # load the camera parameters from the config yaml file
    # import yaml
    # with open('src/camera_calibration/camera_config.yaml', 'r') as file:
    #     config = yaml.safe_load(file)

    # and convert to numpy arrays
    # camera_matrix = torch.tensor(config['camera_matrix'])
    # translation_marix = torch.tensor(config['translation_matrix'])
    # dist_coeffs = torch.tensor(config['dist_coeffs'])

    # generate a random patch
    # first, generate random values between 0 and 1, 
    # then multiply by 255. to receive values between 0. and 255.
    patch = (torch.rand(1, 1, 50, 50) * 255.).requires_grad_()


    # set an arbitrary pose for testing
    # pose includes x,y,z and quaternions in order qx, qy, qz, qw
    # x,y,z will be the center of the patch in camera frame
    # quaternions hold rotation information from attacker (holding the patch) to camera frame
    # pose = torch.tensor([2.7143893241882324,1.6456797122955322,0.4578791558742523, 
    #                      0.0114797880217894, 0.0744068142306854, -0.1520472288581698, 0.985501639095322]).requires_grad_()

    # calculate random translation, rotation (in radians), and scale factor
    tx = torch.randint(high=2, size=(1,)).float().requires_grad_()
    ty = torch.randint(high=2, size=(1,)).float().requires_grad_()
    rotation = torch.distributions.uniform.Uniform(np.radians(-45), np.radians(45)).sample().requires_grad_()  # PyTorch doesn't offer a radians function yet
    scale = torch.distributions.uniform.Uniform(0.01, 1).sample().requires_grad_()
    
    # place the patch
    new_image = place_patch(image, patch, angle=rotation, scale=scale, tx=tx, ty=ty)
    print(new_image.shape)
    # check if image still keeps the gradient
    print(new_image.requires_grad)
    # plot the final image
    plt.imshow(new_image[0][0].detach().numpy())
    plt.show()
