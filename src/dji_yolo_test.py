import torch
import numpy as np
import cv2

# taken from ultralytics yolo
def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

# scale box to original image scale
def scale_box(box, scale_factor_width, scale_factor_height):
    box[0] /= scale_factor_width
    box[1] /= scale_factor_height
    box[2] /= scale_factor_width
    box[3] /= scale_factor_height
    return box

SOFTMAX_SCALE = 20.


if __name__ == '__main__':
  
  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load yolov5
    model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True, autoshape=False)
    model.eval()

    # load image from path with cv2
    img = cv2.imread('misc/dji/images/0000.jpg')
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # scale image to height = 640
    # print(img.shape)
    scale_factor_height = 320 / img.shape[0]
    scale_factor_width = 640 / img.shape[1]
    img_r = cv2.resize(img, (640, 320))

    img_t = torch.tensor(img_r).permute(2, 0, 1).unsqueeze(0).float() / 255.
    img_t = img_t.requires_grad_(True).to(device)
    print(img_t.shape)
    print(img_t.min(), img_t.max())

    # predict boxes
    results = model(img_t)[0]  
    print(results.shape)    # batch_size, num_candidates, boxes + class scores 
    boxes = xywh2xyxy(results[:, :, :4])
    print(boxes.shape)
    scores = results[:, :, 4] * results[0][:, 5] # multiply obj score by person confidence


    soft_scores = torch.nn.functional.softmax(scores * SOFTMAX_SCALE)
    selected_box = torch.bmm(soft_scores.unsqueeze(1), boxes).squeeze(1).detach().cpu().numpy()
    selected_box = scale_box(selected_box[0], scale_factor_width, scale_factor_height).reshape(1, -1)

    print(selected_box)

    # draw box
    xmin, ymin, xmax, ymax = selected_box[0]
    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 10)
    cv2.imwrite('misc/dji/output_softmax.jpg', img)
    
    # top_scores, top_indices = torch.topk(scores, 5)
    # print(top_scores, top_scores.shape)
    # print(top_indices, top_indices.shape)

    # top_indices = top_indices.squeeze(0)

    # top_box = boxes[0, top_indices[0, 0]].detach().cpu().numpy() # img 0, top candidate
    # print(top_box)

    # select boxes
    # top_boxes = boxes[top_indices[0, 0]].detach().cpu().numpy()
    # print(top_boxes)

    # # draw boxes
    # for idx in top_indices:
    #     top_box = boxes[0, idx].detach().cpu().numpy()
    #     scaled_box = scale_box(top_box, scale_factor_width, scale_factor_height)
    #     xmin, ymin, xmax, ymax = scaled_box
    #     cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 10)

    # cv2.imwrite('misc/dji/output.jpg', img)