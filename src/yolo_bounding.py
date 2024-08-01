import torch
import numpy as np
import cv2
import sys

sys.path.append('./yolov5')
print(f'sys.path:')
print()
for string in sys.path:
    print(string) 

import torch.nn as nn
# from google.colab.patches import cv2_imshow
# from models.yolo import Model
# from utils.general import non_max_suppression
from torchvision.ops import box_iou
# from models.common import AutoShape
from util import load_dataset
from torchvision.ops import generalized_box_iou_loss

from camera import Camera


USE_TENSOR = True
USE_AUTOSHAPE = False
TENSOR_DEFAULT_WIDTH = 640
DEBUG = False

import torch
import torch.nn.functional as F

BATCH_SIZE = 1
IMSIZE = (96, 160)

def gen_mask_coords(n, size):
    points = np.random.randint([0, 0], [IMSIZE[0] - size[0], IMSIZE[1]-size[1]], size=(1, 2))
    return torch.tensor([([y, x, y+size[0], x+size[1]]) for (y, x) in points])


def place_patch(images, patch, target):
    batch_size, channels, height, width = images.shape
    patch_height, patch_width = patch.shape[-2:]

    output = torch.zeros_like(images)

    # grid_sample()
    mask = torch.zeros_like(images)
    # Place the patch in the padded_patch tensor
    output[:, :, target[0]:target[0]+patch_height, target[1]:target[1]+patch_width] = patch

    # Update the mask to indicate where the patch is placed
    mask[:, :, target[0]:target[0]+patch_height, target[1]:target[1]+patch_width] = 1

    # print('mask', mask)
    # Combine the images and patches using the mask
    output = (1 - mask) * images + mask * output

    return output


class YOLOBox(nn.Module):
    conf = 0.4  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    softmax_mult = 15.

    def __init__(self, model_config="yolov5/models/yolov5n.yaml", model_ckpt="yolov5n.pt"):
        super().__init__()

        # model = Model(model_config)
        # ckpt = torch.load(model_ckpt)
        # model.load_state_dict(ckpt['model'].state_dict())
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5n", autoshape=False)

        self.model.eval()
        self.c = 0

    def forward(self, og_imgs, show_imgs=False):
        # print("==========FORWARD PASS=========")
        imgs = og_imgs / 255.0

        imgs = torch.repeat_interleave(imgs, 3, dim=1)

        resized_inputs = torch.nn.functional.interpolate(imgs, size=(TENSOR_DEFAULT_WIDTH//2, TENSOR_DEFAULT_WIDTH), mode="bilinear")
        # print("input size", resized_inputs.size())
        # print('resized grad', resized_inputs.grad_fn)
        # tensor w size (B, C, H, W)

        output = self.model(resized_inputs)
        # print("output", type(output), output[0].shape)

        scale_factor = imgs.size()[3] / TENSOR_DEFAULT_WIDTH
        # print('scale', scale_factor)

        # print('output', output[0].grad_fn, output[0].shape)

        # boxes = handle_tensor_output(output, scale_factor)

        # batch_size, num_boxes = output[0].shape[0], output[0].shape[1]
        boxes, scores = self.extract_boxes_and_scores(output[0])

        # print('boxes', boxes.grad_fn, boxes.shape)

        # print('scores shape', scores.shape)

        soft_scores = F.softmax(scores * (15.), dim=1)
        soft_scores = soft_scores.unsqueeze(1)
        selected_boxes = torch.bmm(soft_scores, boxes)


        # debugging
        highest_score_idxs = torch.argmax(scores, 1)
        if show_imgs:
            for i in range(10):
                og_img = og_imgs[i].clone().detach().cpu().numpy()
                og_img = np.moveaxis(og_img, 0, -1)
                og_img = cv2.cvtColor(og_img,cv2.COLOR_GRAY2RGB)

                best_box_score = scores[i, highest_score_idxs[i]]
                true_best_box = boxes[i, highest_score_idxs[i]] * scale_factor

                if DEBUG:
                    print("best box confidence", best_box_score)
                    print(f'best box for image {i}', true_best_box)

                xmin, ymin, xmax, ymax = true_best_box.detach().cpu().numpy().astype(int)
                print(xmin, ymin, xmax, ymax)
                cv2.rectangle(og_img, (xmin, ymin), (xmax, ymax), (255, 0, 0.), 1)

                selected_box = selected_boxes[i][0] * scale_factor
                xmin, ymin, xmax, ymax = int(selected_box[0]), int(selected_box[1]), int(selected_box[2]), int(selected_box[3])
                cv2.rectangle(og_img, (xmin, ymin), (xmax, ymax), (255, 0, 255.), 1)

                cv2.imwrite(f'person_new_{i}.png', og_img)

        return selected_boxes.squeeze()
    
    def extract_boxes_and_scores(self, yolo_output):
        # Extract bounding boxes and scores from YOLO output
        # This function will be specific to the YOLO model's output format

        boxes = xywh2xyxy(yolo_output[:, :, :4])
        # boxes = yolo_output[:, :, :4]  # assuming the first 4 values are the box coordinates
        scores = yolo_output[:, :, 4] *  yolo_output[:, :, 5] # multiply obj score by person confidence
        return boxes, scores


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def generate_tensor(og_img):

    img = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY) / 255.
    # grayscale
    img = np.expand_dims(img, 0)
    img = np.repeat(img, 3, 0)
    print('img shape', np.shape(img))

    # fake batch
    imgs = np.expand_dims(img, 0)
    imgs = np.repeat(imgs, BATCH_SIZE, 0)

    # Run inference
    print('imgs shape', np.shape(imgs))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_img = torch.from_numpy(imgs).to(device).float()
    input_img.requires_grad = True
    resized_inputs = torch.nn.functional.interpolate(input_img, size=(TENSOR_DEFAULT_WIDTH//2, TENSOR_DEFAULT_WIDTH), mode="bilinear")
    print('reszied', resized_inputs.size())
    return input_img

def handle_tensor_output(output, scale_factor):
    # print('output size', len(output))
    # print('output', type(output[0]))
    print(output[0].size())

    y = non_max_suppression(
                        output[0],
                        0.5,
                        0.5,
                        [0],
                        False,
                        False,
                        max_det=2,
                    )

    res = [data * scale_factor for data in y]
    bboxes = []
    for img in res:
        img_boxes = []
        for data in img:
            img_boxes.append((int(data[0]), int(data[1]), int(data[2]), int(data[3])))
        bboxes.append(img_boxes)

    return bboxes

def training_loop():
    batch_size = 16
    lr = 3e-2
    patch_size = (50, 50)
    epochs = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
    train_dataloader = load_dataset(path=dataset_path, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

    target = gen_mask_coords(1, patch_size).to(device)
    print("target:", target)

    wrapper_model = YOLOBox()
    
    patch = torch.rand(1, 1, *patch_size).to(device) * 255.
    patch.requires_grad_(True)
    opt = torch.optim.Adam([patch], lr=lr)
    # loss = 0
    loss = torch.tensor(0.).to(device)
    loss.requires_grad_(True)

    # loss.register_hook(lambda grad: print("hook grad", grad))
    # targets = gen_mask_coords(batch_size, patch_size)

    # for data, labels in train

    for i in range(epochs):
        print("\n")
        print(f' ===== epoch {i} =====  ')
        
        for step, (data, labels) in enumerate(train_dataloader):
            if step%10 == 0:
                print(f'\n step {step}')
            targets = target.expand(data.shape[0], -1)
            data = data.to(device)

            mod_imgs = place_patch(data, patch, target[0])
            # print("mod imgs shape", mod_imgs.shape)

            # for i, data in enumerate(train_features):
            pred_box = wrapper_model(mod_imgs, step%100==0)
            loss = generalized_box_iou_loss(pred_box, targets)

            if step%10 == 0:
                print(loss.sum())
                print(patch.grad)
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
        


def test_dataset():
    model = Model("yolov5/models/yolov5s.yaml")
    ckpt = torch.load("yolov5s.pt")
    model.load_state_dict(ckpt['model'].state_dict())

    dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
    train_dataloader = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=False, num_workers=0)

    if USE_AUTOSHAPE:   
        wrapper_model = AutoShape(model)
    else:
        wrapper_model = YOLOBox()
    wrapper_model.classes = [0]

    # Inference on images
    path = "./person.png"
    og_img = cv2.imread(path)
    
    height, width, _ = np.shape(og_img)
    print("height width", height, width)

    for i in range(5):
        print('\n')
        print(f'FORWARD PASS ITERATION {i}')
        train_features, train_labels = next(iter(train_dataloader))
        if USE_TENSOR:
            if USE_AUTOSHAPE:
                input_img = generate_tensor(og_img)
                # input_img = train_features
            else:
                input_img = train_features
                # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # input_img = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY) 
                # input_img = np.expand_dims(input_img, 0)
                # input_img = np.expand_dims(input_img, 0)
                # input_img = np.repeat(input_img, BATCH_SIZE, 0)
                # input_img = torch.from_numpy(input_img).to(device).float()
            
        else:
            input_img = og_img

        # wrapper_model.max_det = 100
        wrapper_model.classes = [0]
        wrapper_model.conf = 0.4
        wrapper_model.iou = 0.5
        # wrapper_model.eval()
        

        if USE_TENSOR:
            output = wrapper_model(input_img)

def test_image():
    model = Model("yolov5/models/yolov5s.yaml")
    ckpt = torch.load("yolov5s.pt")
    model.load_state_dict(ckpt['model'].state_dict())
    cam_config = 'misc/camera_calibration/calibration.yaml'
    cam = Camera(cam_config)

    if USE_AUTOSHAPE:   
        wrapper_model = AutoShape(model)
    else:
        wrapper_model = YOLOPoseDetection(cam)
    wrapper_model.classes = [0]

    # Inference on images
    path = "./person.png"
    og_img = cv2.imread(path)
    
    height, width, _ = np.shape(og_img)
    print("height width", height, width)


    if USE_TENSOR:
        if USE_AUTOSHAPE:
            input_img = generate_tensor(og_img)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_img = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY) 
            input_img = np.expand_dims(input_img, 0)
            input_img = np.expand_dims(input_img, 0)
            input_img = np.repeat(input_img, BATCH_SIZE, 0)
            input_img = torch.from_numpy(input_img).to(device).float()
        
    else:
        input_img = og_img

    # wrapper_model.max_det = 100
    wrapper_model.classes = [0]
    wrapper_model.conf = 0.4
    wrapper_model.iou = 0.5
    # wrapper_model.eval()
    

    if USE_TENSOR:
        output = wrapper_model(input_img)

def test_place_patch():
    images = torch.ones((3, 1, 8, 8))
    patch = torch.ones((2, 2)) * 2.0
    target = torch.tensor([1, 1, 3, 3])

    mod_img = place_patch(images, patch, target)
    print(mod_img)

if __name__ == "__main__":
    # test_image()
    # test_dataset()
    training_loop()
    
    
    


# highest_score_idxs = torch.argmax(scores, 1)

        # total_err = 0
        # best_boxes = torch.zeros((batch_size, 4))

        # for i in range(batch_size):
        #     if DEBUG:
        #         print(f'image {i}')

        #     # og_img = og_imgs[i].clone().detach().numpy()
        #     # og_img = np.moveaxis(og_img, 0, -1)
        #     # og_img = cv2.cvtColor(og_img,cv2.COLOR_GRAY2RGB)

        #     best_box_score = scores[i, highest_score_idxs[i]]
        #     # true_best_box = boxes[i, highest_score_idxs[i]] * scale_factor

        #     if DEBUG:
        #         print("best box confidence", best_box_score)
        #         # print(f'best box for image {i}', true_best_box)

        #     # xmin, ymin, xmax, ymax = true_best_box.detach().numpy().astype(int)
        #     # print(xmin, ymin, xmax, ymax)
        #     # cv2.rectangle(og_img, (xmin, ymin), (xmax, ymax), (255, 0, 0.), 1)

        #     if best_box_score > 0.2:

        #         soft_scores = F.softmax(scores[i] * (15.), dim=0)  # Shape: (N,)

        #         # Compute the weighted sum of the boxes
        #         selected_box = torch.sum(boxes[i] * soft_scores.unsqueeze(1), dim=0)  # Shape: (4,)
        #         selected_box = (selected_box) * scale_factor

        #         print('selected', selected_box.grad_fn)

        #         if DEBUG:
        #             print('selected box', selected_box)

        #         # xmin, ymin, xmax, ymax = int(selected_box[0]), int(selected_box[1]), int(selected_box[2]), int(selected_box[3])
        #         # cv2.rectangle(og_img, (xmin, ymin), (xmax, ymax), (255, 0, 255.), 1)

        #         # err = torch.norm(true_best_box - selected_box)
        #         # if DEBUG:
        #         #     print('err', err)
        #         #     if err > 5:
        #         #         print("BIG ERROR")

        #         best_boxes[i] = selected_box

        #         # total_err += err.item()
            
        #     # print('\n')
        #     # cv2.imwrite(f'iter_{self.c}_person_new_{i}.png', og_img)

        # print('total err', total_err/batch_size)

        # # print('best', best_boxes.grad_fn)
        # self.c += 1
        # return best_boxes