import torch
import numpy as np
import cv2
import sys

sys.path.append('src/')
# print(f'sys.path:')
# print()
# for string in sys.path:
#     print(string) 

import torch.nn as nn
from util import load_dataset, printd
from torchvision.ops import generalized_box_iou_loss

from camera import Camera


USE_TENSOR = True
USE_AUTOSHAPE = False
TENSOR_DEFAULT_WIDTH = 640
DEBUG = True

import torch
import torch.nn.functional as F

BATCH_SIZE = 1
IMSIZE = (96, 160)
SOFTMAX_MULT = 15.

# get random coordinates for where patch should be
def gen_patch_coords(n, size):
    points = np.random.randint([0, 0], [IMSIZE[0] - size[0], IMSIZE[1]-size[1]], size=(1, 2))
    return torch.tensor([([y, x, y+size[0], x+size[1]]) for (y, x) in points])


class YOLOBox(nn.Module):
    conf = 0.4  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    softmax_mult = 15.

    def __init__(self, cam_config='misc/camera_calibration/calibration.yaml'):
        super().__init__()

        # load model
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5n", autoshape=False)
        self.model.eval()

        # camera
        self.cam = Camera(cam_config)

    def forward(self, og_imgs, show_imgs=False):
        imgs = og_imgs / 255.0

        imgs = torch.repeat_interleave(imgs, 3, dim=1)

        # yolo wants size (320, 640)
        resized_inputs = torch.nn.functional.interpolate(imgs, size=(TENSOR_DEFAULT_WIDTH//2, TENSOR_DEFAULT_WIDTH), mode="bilinear")
        output = self.model(resized_inputs)

        scale_factor = imgs.size()[3] / TENSOR_DEFAULT_WIDTH
        boxes, scores = self.extract_boxes_and_scores(output[0])

        # take a weighted average of the boxes
        soft_scores = F.softmax(scores * SOFTMAX_MULT, dim=1)
        soft_scores = soft_scores.unsqueeze(1)
        selected_boxes = torch.bmm(soft_scores, boxes) * scale_factor

        # printd('selected ', selected_boxes.shape, selected_boxes.grad_fn)

        # debugging
        if show_imgs:
            # true best boxes
            highest_score_idxs = torch.argmax(scores, 1)

            for i in range(min(len(og_imgs), 10)):
                # print(og_imgs.shape)
                og_img = og_imgs[i].clone().detach().cpu().numpy()
                og_img = np.moveaxis(og_img, 0, -1)
                og_img = cv2.cvtColor(og_img,cv2.COLOR_GRAY2RGB)

                true_best_box = boxes[i, highest_score_idxs[i]] * scale_factor

                xmin, ymin, xmax, ymax = true_best_box.detach().cpu().numpy().astype(int)
                cv2.rectangle(og_img, (xmin, ymin), (xmax, ymax), (255, 0, 0.), 1)

                selected_box = selected_boxes[i][0]
 
                xmin, ymin, xmax, ymax = int(selected_box[0]), int(selected_box[1]), int(selected_box[2]), int(selected_box[3])
                cv2.rectangle(og_img, (xmin, ymin), (xmax, ymax), (255, 0, 255.), 1)

                cv2.imwrite(f'person_new_{i}.png', og_img)

        xyzs = self.cam.batch_xyz_from_boxes(selected_boxes.squeeze(1))  # only using squeeze() here will cause all dimensions to be deleted if there's only one input image
        return xyzs
    
    def extract_boxes_and_scores(self, yolo_output):
        # Extract bounding boxes and scores from YOLO output
        # This function will be specific to the YOLO model's output format

        boxes = self.xywh2xyxy(yolo_output[:, :, :4])
        scores = yolo_output[:, :, 4] *  yolo_output[:, :, 5] # multiply obj score by person confidence
        return boxes, scores

    # taken from ultralytics yolo
    def xywh2xyxy(self, x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y



# ========== stuff for testing =========== 


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

    target = gen_patch_coords(1, patch_size).to(device)
    print("target:", target)

    wrapper_model = YOLOBox()
    
    patch = torch.rand(1, 1, *patch_size).to(device) * 255.
    patch.requires_grad_(True)
    opt = torch.optim.Adam([patch], lr=lr)

    loss = torch.tensor(0.).to(device)
    loss.requires_grad_(True)

    for i in range(epochs):
        print("\n")
        print(f' ===== epoch {i} =====  ')
        
        for step, (data, labels) in enumerate(train_dataloader):
            if step%10 == 0:
                print(f'\n step {step}')
            targets = target.expand(data.shape[0], -1)
            data = data.to(device)

            mod_imgs = place_patch(data, patch, target[0])
            # mod_imgs = data
            # print("mod imgs shape", mod_imgs.shape)

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
    pass
    