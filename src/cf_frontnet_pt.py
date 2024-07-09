import torch
import os, sys
import cv2
import numpy as np


sys.path.append('pulp-frontnet/PyTorch/Frontnet')

from Frontnet import FrontnetModel
from DataProcessor import DataProcessor
from Dataset import Dataset
from torch.utils import data

import rowan

class CFSim():
    def __init__(self, model_path="pulp-frontnet/PyTorch/Models/Frontnet160x32.pt", dataset_path="simulators/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        pytorch_model = self.load_model(model_path, self.device)
        self.pose_estimator = pytorch_model
        
        self.pose = np.array([0., 0., 0., 0.]) # x, y, z, yaw
        
        self.current_idx = 0
        # self.target_trajectory = target_trajectory

        # might be deleted later, the dataset is only loaded to get a suitable background image
        # self.dataset, _ = self.load_dataset(dataset_path)
        # base_img, gt = self.dataset.dataset.__getitem__(0)
        # self.base_img = base_img#.squeeze(0).numpy()

        # patch stays random for now and inside the simulator for compatibility with current
        # optimize script
        self.patch = np.random.rand(10, 10, 1).astype(np.float32) * 255. # load one of the optimized FAPs instead!

    def load_model(self, path, device, config="160x32"):
        """
        From FAP repo
        Loads a saved Frontnet model from the given path with the set configuration and moves it to CPU/GPU.
        Parameters
            ----------
            path
                The path to the stored Frontnet model
            device
                A PyTorch device (either CPU or GPU)
            config
                The architecture configuration of the Frontnet model. Must be one of ['160x32', '160x16', '80x32']
        """
        assert config in FrontnetModel.configs.keys(), 'config must be one of {}'.format(list(FrontnetModel.configs.keys()))
        
        # get correct architecture configuration
        model_params = FrontnetModel.configs[config]
        # initialize a random model with configuration
        model = FrontnetModel(**model_params).to(device)
        
        # load the saved model 
        try:
            model.load_state_dict(torch.load(path, map_location=device)['model'])
        except RuntimeError:
            print("RuntimeError while trying to load the saved model!")
            print("Seems like the model config does not match the saved model architecture.")
            print("Please check if you're loading the right model for the chosen config!")

        return model.eval()
    
    # only needed during testing
    def load_dataset(self, path, batch_size = 32, shuffle = False, drop_last = True, num_workers = 1, train=True, train_set_size=0.9):
        # From FAP repo
        # load images and labels from the stored dataset
        [images, labels] = DataProcessor.ProcessTestData(path)

        
        # init RNG for loading the data always with the same key to
        # ensure the same images end up in train and test set respectively
        rng = np.random.default_rng(1749)

        # split dataset into train and test set
        indices = np.arange(len(images))
        rng.shuffle(indices)
        split_idx = int(len(images) * train_set_size)

        train_set = Dataset(images[indices[:split_idx]], labels[indices[:split_idx]])
        test_set = Dataset(images[indices[split_idx:]], labels[split_idx:])

        # for quick and convinient access, create a torch DataLoader with the given parameters
        data_params = {'batch_size': batch_size, 'shuffle': shuffle, 'drop_last':drop_last, 'num_workers': num_workers}
        train_loader = data.DataLoader(train_set, **data_params)
        test_loader = data.DataLoader(test_set, **data_params)

        return train_loader, test_loader


    def _perspective_grid(self,
    coeffs: [float], 
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
        
    def project_patch(self, patch, T, image):
        # using cv2 to project the patch instead of FAP place_patch() function,
        # since we don't need to calculate gradients
        width, height = image.shape[:2]
        # print(height, width)
        mask = np.ones_like(patch)

        warped_patch = cv2.warpPerspective(patch, T, (height, width), flags=cv2.INTER_NEAREST)
        mask = cv2.warpPerspective(mask, T, (height, width), flags=cv2.INTER_NEAREST)

        mod_img = image * ~mask.astype(bool)
        mod_img += warped_patch

        return mod_img # return a np array instead of jnp array and convert to double

        # sanity check with torch perspective grid
    def pt_project_patch(self, patch, T, base_img):
        patch_t = torch.tensor(patch, dtype=torch.float64, device=self.device).unsqueeze(0).unsqueeze(0)
        img = torch.tensor(base_img, device=self.device)

        p_height, p_width = patch.shape[-2:]
        i_height, i_width = img.shape[-2:]

        mask = torch.ones_like(patch_t, dtype=torch.float64, device=self.device)

        inv_t = np.linalg.inv(T)
        coeffs = torch.tensor(np.array([inv_t.flatten()]), dtype=torch.float64, device=self.device)
        
        grid = self._perspective_grid(coeffs, w=p_width, h=p_height, dtype=torch.float64, ow=i_width, oh=i_height, device=self.device, center = [1., 1.])

        bit_mask = torch.nn.functional.grid_sample(mask, grid, mode='bilinear', align_corners=False, padding_mode='zeros').bool()
        transformed_patch = torch.nn.functional.grid_sample(patch_t, grid, mode='bilinear', align_corners=False, padding_mode='zeros')

        modified_image = img * ~bit_mask.bool()
        modified_image += transformed_patch

        return modified_image
    
    def sim_new_pose(self, image: np.ndarray):
        # make sure images (plural if working with batches) are of correct shape
        assert np.prod(image.shape) % (96 * 160) == 0 
        image_t = torch.tensor(image).to(self.device)
        if len(image_t.shape) < 3:
            image_t = image_t.unsqueeze(0).unsqueeze(0)
        if len(image_t.shape) == 3:
            image_t = image_t.unsqueeze(1)
        
        # frontnet prediction
        x, y, z, yaw = self.pose_estimator(image_t)
        # reshape output and turn into numpy array
        predicted_pose = torch.hstack((x, y, z, yaw))
        predicted_pose = predicted_pose.detach().cpu().numpy()

        # print("predicted pose", predicted_pose, predicted_pose.shape)
        
        # calculate controller output
        new_setpoint = self._controller_setpoint(predicted_pose)

        return new_setpoint

    def _controller_setpoint(self, predicted_poses):
        setpoints = []
        for predicted_pose in predicted_poses:
            quats = rowan.from_euler(0., 0., self.pose[3], convention='xyz') # returns qw, qx, qy, qz
            rotated_desired = rowan.rotate(quats, predicted_pose[:3])
            target_pos = predicted_pose[:3] + rotated_desired

            # predicted yaw angles are discarded since they are very faulty
            global_pos = target_pos - self.pose[:3]
            target_yaw = np.arctan2(global_pos[1], global_pos[0]) - np.pi

            new_setpoint = target_pos + self._calc_heading_vec(1., target_yaw)
            setpoints.append([*new_setpoint, target_yaw])
        
        return np.array(setpoints)

    def _calc_heading_vec(self, radius, angle):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return np.array([x, y, 0.0])

    def _pred_to_numpy(self, prediction):
        x, y, z, yaw = prediction
        x = x.detach().cpu().squeeze(0).squeeze(0).numpy()
        y = y.detach().cpu().squeeze(0).squeeze(0).numpy()
        z = z.detach().cpu().squeeze(0).squeeze(0).numpy()
        yaw = yaw.detach().cpu().squeeze(0).squeeze(0).numpy()

        return np.array([x, y, z, yaw])

    def update(self, pose):
        self.pose = pose
        self.current_idx += 1

    def reset(self):
        self.pose = np.array([0., 0., 0.])
        self.current_idx = 0

    def eval(self, pose, desired_pose):
        l2_distances = np.linalg.norm((pose - desired_pose), ord=2)#, axis=1)
        return l2_distances
    

if __name__ == '__main__':

    model_path = "pulp-frontnet/PyTorch/Models/Frontnet160x32.pt"
    dataset_path = "pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"

    # won't be necessary later
    # sys.path.append('.')
    # from util import bezier_curve
    # control_points_bezier = np.array([[0, 0, 0.0], [1, 3, 0.4], [2, -1, 0.8], [3, 2, 1]])
    # target_trajectory = np.array(bezier_curve(control_points_bezier, 20))

    cf_sim = CFSim(model_path, dataset_path)

    # # test with single image
    # base_img = cf_sim.base_img.unsqueeze(0).to(cf_sim.device)
    # out_pytorch = torch.hstack(cf_sim.pose_estimator(base_img))
    # print("Output frontnet: ", out_pytorch)
    # new_pose = cf_sim.sim_new_pose(cf_sim.base_img)
    # print("Output controller: ", new_pose)

    # # test with batch
    # print("Test with batch")
    # base_imgs = cf_sim.base_img.repeat(2, 1, 1, 1).to(cf_sim.device)
    # x, y, z, yaw = cf_sim.pose_estimator(base_imgs)
    # out_pytorch = torch.hstack(cf_sim.pose_estimator(base_imgs))
    # print("Output frontnet: ", out_pytorch)
    # new_pose = cf_sim.sim_new_pose(base_imgs)
    # print("Output controller: ", new_pose)

    patch = np.random.rand(3,3) * 255.
    base_img = cf_sim.base_img[0]

    print(patch.shape)
    print(base_img.shape)

    T = np.eye(3,3)   # basic transformation matrix
    T[0, 0] = T[1, 1] = 10. # scale factor
    # patch upper left corner at center of image
    T[0, 2] = 80.
    T[1, 2] = 48.

    mod_img = cf_sim.project_patch(patch, T, base_img).to(cf_sim.device)
    mod_img = mod_img.unsqueeze(0).unsqueeze(0)
    print(mod_img.shape, mod_img.dtype)

    print(cf_sim.pose_estimator(mod_img))
    # out_pytorch = torch.hstack(cf_sim.pose_estimator(mod_img.unsqueeze(0).unsqueeze(0)))
    # print("Output frontnet: ", out_pytorch)
    # new_pose = cf_sim.sim_new_pose(mod_img)
    # print("Output controller: ", new_pose)

    # from matplotlib import pyplot as plt
    # plt.imshow(mod_img, cmap='gray')
    # plt.show()