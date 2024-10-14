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
from models.yolo import Model
from utils.general import non_max_suppression
from torchvision.ops import box_iou
from models.common import AutoShape
from util import load_dataset

from camera import Camera


USE_TENSOR = True
USE_AUTOSHAPE = False
TENSOR_DEFAULT_WIDTH = 640

import torch
import torch.nn.functional as F

BATCH_SIZE = 1


class YOLOPoseDetection(nn.Module):
    conf = 0.4  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    softmax_mult = 15.

    def __init__(self, cam, model_config="yolov5/models/yolov5n.yaml", model_ckpt="yolov5n.pt"):
        super().__init__()

        model = Model(model_config)
        ckpt = torch.load(model_ckpt)
        model.load_state_dict(ckpt['model'].state_dict())

        self.model = model.eval()
        self.cam = cam
        self.c = 0

    def forward(self, og_imgs):
        print("==========FORWARD PASS=========")
        imgs = og_imgs / 255.0

        imgs = torch.repeat_interleave(imgs, 3, dim=1)
        imgs.requires_grad_()

        resized_inputs = torch.nn.functional.interpolate(imgs, size=(TENSOR_DEFAULT_WIDTH//2, TENSOR_DEFAULT_WIDTH), mode="bilinear")
        print("input size", resized_inputs.size())
        # tensor w size (B, C, H, W)

        output = self.model(resized_inputs)
        # return output

        scale_factor = imgs.size()[3] / TENSOR_DEFAULT_WIDTH
        print('scale', scale_factor)

        boxes = handle_tensor_output(output, scale_factor)
        # print('CORRECT BOXES:', boxes)
        # print("\n")

        # for box in boxes:
        #     xmin, ymin, xmax, ymax = box
        #     cv2.rectangle(og_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
        # cv2.imwrite('person_new.png', og_img)


        batch_size, num_boxes = output[0].shape[0], output[0].shape[1]
        boxes, scores = self.extract_boxes_and_scores(output[0])
        print('scores shape', scores.shape)

        highest_score_idxs = torch.argmax(scores, 1)

        total_err = 0

        for i in range(batch_size):
            print(f'image {i}')
            og_img = og_imgs[i].detach().numpy()
            # print('og img before', np.shape(og_img))
            og_img = np.moveaxis(og_img, 0, -1)
            # print('og img', np.shape(og_img))
            og_img = cv2.cvtColor(og_img,cv2.COLOR_GRAY2RGB)

            best_box_score = scores[i, highest_score_idxs[i]]
            print("best box confidence", best_box_score)
            true_best_box = boxes[i, highest_score_idxs[i]] * scale_factor
            print(f'best box for image {i}', true_best_box)

            xmin, ymin, xmax, ymax = true_best_box.detach().numpy().astype(int)
            # print(xmin, ymin, xmax, ymax)
            # cv2.rectangle(og_img, (xmin, ymin), (xmax, ymax), (255, 0, 0.), 1)

            if best_box_score > 0.2:

                soft_scores = F.softmax(scores[i] * (15.), dim=0)  # Shape: (N,)

                # Compute the weighted sum of the boxes
                selected_box = torch.sum(boxes[i] * soft_scores.unsqueeze(1), dim=0)  # Shape: (4,)
                selected_box = (selected_box) * scale_factor

                print('selected box', selected_box)
                xmin, ymin, xmax, ymax = int(selected_box[0]), int(selected_box[1]), int(selected_box[2]), int(selected_box[3])
                # cv2.rectangle(og_img, (xmin, ymin), (xmax, ymax), (255, 0, 255.), 1)

                # print(true_best_box - selected_box)
                err = torch.norm(true_best_box - selected_box)
                print('err', err)
                if err > 5:
                    print("BIG ERROR")

                total_err += err.item()

            
            print('\n')
            cv2.imwrite(f'iter_{self.c}_person_new_{i}.png', og_img)
        
        print('total err', total_err/batch_size)
        self.c += 1
    
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
    print('output size', len(output))
    print('output', type(output[0]))
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

    # print("type y", type(y))
    # y is a list of len = batch_size
    # res = y[0] * scale_factor
    res = [data * scale_factor for data in y]

    # print("y0", len(y), y[0], y[0].size())
    # print('res', res)

    # best_bbox = res[0]
    bboxes = []
    for img in res:
        img_boxes = []
        for data in img:
            img_boxes.append((int(data[0]), int(data[1]), int(data[2]), int(data[3])))
        bboxes.append(img_boxes)

    return bboxes


def test_dataset():
    model = Model("yolov5/models/yolov5s.yaml")
    ckpt = torch.load("yolov5s.pt")
    model.load_state_dict(ckpt['model'].state_dict())
    cam_config = 'misc/camera_calibration/calibration.yaml'
    cam = Camera(cam_config)

    dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
    train_dataloader = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=False, num_workers=0)

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

    for i in range(1):
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

if __name__ == "__main__":
    # test_image()
    test_dataset()
    