import torch
import sys

sys.path.insert(0,'pulp-frontnet/PyTorch/')
from Frontnet.Frontnet import FrontnetModel

from Frontnet.DataProcessor import DataProcessor
from Frontnet.Dataset import Dataset
from torch.utils import data

import rowan

# import nemo
# from Frontnet.Utils import ModelManager

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
# import random

import onnx
from onnx import numpy_helper

import glob

DEBUG_GRAD = False

def printd(*args, **kwargs):
    if DEBUG_GRAD:
        print(*args, **kwargs)

def load_model(path, device, config):
    """
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
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True)['model'])
    except RuntimeError:
        print("RuntimeError while trying to load the saved model!")
        print("Seems like the model config does not match the saved model architecture.")
        print("Please check if you're loading the right model for the chosen config!")

    return model

def load_quantized(path, device):
    model = FrontnetQuantizedModel(path, device)

    return model

class FrontnetQuantizedModel(torch.nn.Module):
    def __init__(self, path, device):
        super(FrontnetQuantizedModel, self).__init__()

        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)

        self.weights = {}
        for node in onnx_model.graph.initializer:
            data = torch.tensor(numpy_helper.to_array(node)).to(device)
            if 'kappa' in node.name:
                name = node.name.replace('kappa', 'gamma')
            elif 'lamda' in node.name:
                name = node.name.replace('lamda', 'beta')
            else:
                name = node.name
            self.weights.update({name: data})
        

        constants = []
        for node in onnx_model.graph.node:
            if node.op_type == "Constant":
                const_value = numpy_helper.to_array(node.attribute[0].t)
                constants.append(const_value.item())
        self.constants = torch.tensor(np.array(constants)).to(device)
        

    def forward(self, x):
        # layer 0
        conv0 = torch.nn.functional.conv2d(x, self.weights['conv.weight'], stride=2, padding=2, bias=None)
        bn0 = self.weights['bn.gamma'] * conv0 + self.weights['bn.beta']
        mul0 = bn0 * self.constants[0]
        div0 = mul0 / self.constants[1]
        rel0 = torch.nn.functional.relu(div0)
        max0 = torch.nn.functional.max_pool2d(rel0, kernel_size=2, stride=2, padding=0)


        # layer 1
        conv1_1 = torch.nn.functional.conv2d(max0, self.weights['layer1.conv1.weight'], stride=2, padding=1, bias=None)
        bn1_1 = self.weights['layer1.bn1.gamma'] * conv1_1 + self.weights['layer1.bn1.beta']
        mul1_1 = bn1_1 * self.constants[2]
        div1_1 = mul1_1 / self.constants[3]
        rel1_1 = torch.nn.functional.relu(div1_1)

        conv1_2 = torch.nn.functional.conv2d(rel1_1, self.weights['layer1.conv2.weight'], stride=1, padding=1, bias=None)
        bn1_2 = self.weights['layer1.bn2.gamma'] * conv1_2 + self.weights['layer1.bn2.beta']
        mul1_2 = bn1_2 * self.constants[4]
        div1_2 = mul1_2 / self.constants[5]
        rel1_2 = torch.nn.functional.relu(div1_2)

        # layer 2
        conv2_1 = torch.nn.functional.conv2d(rel1_2, self.weights['layer2.conv1.weight'], stride=2, padding=1, bias=None)
        bn2_1 = self.weights['layer2.bn1.gamma'] * conv2_1 + self.weights['layer2.bn1.beta']
        mul2_1 = bn2_1 * self.constants[6]
        div2_1 = mul2_1 / self.constants[7]
        rel2_1 = torch.nn.functional.relu(div2_1)

        conv2_2 = torch.nn.functional.conv2d(rel2_1, self.weights['layer2.conv2.weight'], stride=1, padding=1, bias=None)
        bn2_2 = self.weights['layer2.bn2.gamma'] * conv2_2 + self.weights['layer2.bn2.beta']
        mul2_2 = bn2_2 * self.constants[8]
        div2_2 = mul2_2 / self.constants[9]
        rel2_2 = torch.nn.functional.relu(div2_2)

        # layer 3
        conv3_1 = torch.nn.functional.conv2d(rel2_2, self.weights['layer3.conv1.weight'], stride=2, padding=1, bias=None)
        bn3_1 = self.weights['layer3.bn1.gamma'] * conv3_1 + self.weights['layer3.bn1.beta']
        mul3_1 = bn3_1 * self.constants[10]
        div3_1 = mul3_1 / self.constants[11]
        rel3_1 = torch.nn.functional.relu(div3_1)

        conv3_2 = torch.nn.functional.conv2d(rel3_1, self.weights['layer3.conv2.weight'], stride=1, padding=1, bias=None)
        bn3_2 = self.weights['layer3.bn2.gamma'] * conv3_2 + self.weights['layer3.bn2.beta']
        mul3_2 = bn3_2 * self.constants[12]
        div3_2 = mul3_2 / self.constants[13]
        rel3_2 = torch.nn.functional.relu(div3_2)

        flat = rel3_2.flatten(1)
        out = torch.nn.functional.linear(flat, self.weights['fc.weight'], self.weights['fc.bias'])      
        x_q = out[:, 0]
        y_q = out[:, 1]
        z_q = out[:, 2]
        phi_q = out[:, 3]

        # de-quantize results
        x = x_q * 2.46902e-05 + 1.02329e+00
        y = y_q * 2.46902e-05 + 7.05523e-04
        z = z_q * 2.46902e-05 + 2.68245e-01
        phi = phi_q * 2.46902e-05 + 5.60173e-04

        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        z = z.unsqueeze(1)
        phi = phi.unsqueeze(1)

        return [x, y, z, phi]

    # def forward(self, batch):
    #     batch_x = []
    #     batch_y = []
    #     batch_z = []
    #     batch_phi = []
    #     for x in batch:
    #         x, y, z, phi = self._forward_single(x)
    #         batch_x.append(x)
    #         batch_y.append(y)
    #         batch_z.append(z)
    #         batch_phi.append(phi)

    #     return [torch.cat(batch_x), torch.cat(batch_y), torch.cat(batch_z), torch.cat(batch_phi)]

def load_dataset(path, batch_size = 32, shuffle = False, drop_last = True, num_workers = 1, train=True, train_set_size=0.9, IMRC=True):
    """
    Loads a dataset from the given path. 
    Parameters
        ----------
        path
            The path to the dataset
        batch_size
            The size of the batches the dataset will contain
        shuffle
            If set to True, the data will be shuffled randomly
        drop_last
            If set to True, the last batch of the dataset will be dropped. 
            This ensures that all returned batches are of the same size.
        num_workers
            Set the number of workers.
        train
            If set to True, the function will return the train set. If set to 
            False, the test set will be returned instead.
        train_set_size
            Set the ratio of total images included in the train set.
        IMRC
            If set to True, the datasets will include images from our flight 
            space acquired with our camera configuration. Further information
            on dataset acquisition is provided in the README. 
    """
    # load images and labels from the stored dataset
    [images, labels] = DataProcessor.ProcessTestData(path)

    # if training data should be extended by our custom dataset, set IMRC to True
    if IMRC:
        import pickle
        with open("misc/IMRC_images.pickle", "rb") as f:
            imrc_data = pickle.load(f)

        imrc_images = imrc_data['x']
        imrc_labels = imrc_data['y']

        images = np.concatenate([images, imrc_images])
        labels = np.concatenate([labels, imrc_labels])

    rng = np.random.default_rng(1749)

    indices = np.arange(len(images))
    rng.shuffle(indices)
    split_idx = int(len(images) * train_set_size)

    if train:
        # create a torch dataset from the loaded data
        dataset = Dataset(images[indices[:split_idx]], labels[indices[:split_idx]])
    else:
        dataset = Dataset(images[indices[split_idx:]], labels[split_idx:])

    # for quick and convinient access, create a torch DataLoader with the given parameters
    data_params = {'batch_size': batch_size, 'shuffle': shuffle, 'drop_last':drop_last, 'num_workers': num_workers}
    data_loader = data.DataLoader(dataset, **data_params)
    
    return data_loader


def calc_saliency(img, gt, model):
    input = img.requires_grad_(True)
    prediction = torch.stack(model(input.float())).permute(1, 0, 2).squeeze(2).squeeze(0)

    loss_x = torch.nn.L1Loss()(prediction[0], gt[0])
    loss_y = torch.nn.L1Loss()(prediction[1], gt[1])
    loss_z = torch.nn.L1Loss()(prediction[2], gt[2])
    loss_phi = torch.nn.L1Loss()(prediction[3], gt[3])

    loss = loss_x + loss_y + loss_z + loss_phi

    loss.backward()

    saliency = input.grad.data.abs()

    return saliency

def plot_saliency(img, gt, model):
    saliency = calc_saliency(img, gt, model)
    img = img[0][0].detach().cpu().numpy()
    saliency = saliency[0][0].detach().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(8, 2))
    ax[0].imshow(img, cmap='gray')
    ax[1].set_title('Saliency Map')
    ax[1].imshow(saliency, cmap='hot')
    ax[2].set_title('Superimposed')
    ax[2].imshow(img + (200000*saliency), cmap='gray')

    return fig

def inverse_norm(val, minimum, maximum):
    return np.arctanh(2* ((val - minimum) / (maximum - minimum)) - 1)

def get_transformation(sf, tx, ty):
    translation_vector = torch.stack([tx, ty]).unsqueeze(0) #torch.zeros([1], device=tx.device)]).unsqueeze(0)

    eye = torch.eye(2, 2).unsqueeze(0).to(sf.device)
    scale = eye * sf

    # print(scale.shape, translation_vector.shape)

    transformation_matrix = torch.cat([scale, translation_vector], dim=2)
    return transformation_matrix.float()

def norm_transformation(sf, tx, ty):
    tx_tanh = torch.tanh(tx)
    ty_tanh = torch.tanh(ty)
    scaling_norm = 0.1 * (torch.tanh(sf) + 1) + 0.3 # normalizes scaling factor to range [0.3, 0.5]

    return scaling_norm, tx_tanh, ty_tanh


def gen_noisy_transformations(batch_size, sf, tx, ty):
    noisy_transformation_matrix = []
    for i in range(batch_size):
        sf_n = sf + np.random.normal(0.0, 0.1)
        tx_n = tx + np.random.normal(0.0, 0.1)
        ty_n = ty + np.random.normal(0.0, 0.1)

        scale_norm, tx_norm, ty_norm = norm_transformation(sf_n, tx_n, ty_n)
        matrix = get_transformation(scale_norm, tx_norm, ty_norm)

        # random_yaw = np.deg2rad(np.random.normal(-10, 10))
        # random_pitch = np.deg2rad(np.random.normal(-5, 5))
        # random_roll = np.deg2rad(np.random.normal(-5, 5))
        #noisy_rotation = torch.tensor(get_rotation(random_yaw, random_pitch, random_roll)).float().to(matrix.device)

        #matrix[..., :3, :3] = noisy_rotation @ matrix[..., :3, :3]

        noisy_transformation_matrix.append(matrix)
    
    return torch.cat(noisy_transformation_matrix)

def load_patch(path, mode):
    patches = []
    for file_p in sorted(glob.glob(str(path) + '/' + str(mode)+ '*/' + 'patches.npy')):
        patches.append(np.load(file_p)[-1])

    return np.array(patches)

def load_position(path, mode):
    positions = []
    for file_p in sorted(glob.glob(str(path) + '/' + str(mode)+ '*/' + 'positions_norm.npy')):
        positions.append(np.load(file_p))

    positions = np.rollaxis(np.array(positions), 2, 0)[-1]
    positions = np.rollaxis(positions, 2, 1)
    positions = np.rollaxis(positions, 1, 0)

    return positions


# rotation vectors are axis-angle format in "compact form", where
# theta = norm(rvec) and axis = rvec / theta
# they can be converted to a matrix using cv2. Rodrigues, see
# https://docs.opencv.org/4.7.0/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
def opencv2quat(rvec):
    angle = np.linalg.norm(rvec)
    if angle == 0:
        q = np.array([1,0,0,0])
    else:
        axis = rvec.flatten() / angle
        q = rowan.from_axis_angle(axis, angle)
    return q


# def plot_patch(patch, image, title='Plot', save=False, path='./'):

#     img_min, img_max = patch.batch_place(image)

#     f = plt.figure(constrained_layout=True, figsize=(10, 4))
#     subfigs = f.subfigures(1, 2, width_ratios=[1, 3])
#     fig_patch = subfigs[0].subplots(1,1)
#     fig_patch.imshow(patch.patch[0][0].detach().cpu().numpy(), cmap='gray')
#     subfigs[0].suptitle('Patch', fontsize='x-large')

#     subfigs[1].suptitle('placed', fontsize='x-large')
#     fig_placed = subfigs[1].subplots(1,2)
#     fig_placed[0].imshow(img_min[0][0].detach().cpu().numpy(), cmap='gray')
#     fig_placed[0].set_title('min direction')
#     fig_placed[1].imshow(img_max[0][0].detach().cpu().numpy(), cmap='gray')
#     fig_placed[1].set_title('max direction')

#     f.suptitle(title, fontsize='xx-large')
    
#     if save:
#         plt.savefig(path+title+'.jpg', transparent=False)
#         plt.close()
#     else: 
#         return f


