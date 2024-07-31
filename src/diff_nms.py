import torch
import torch.nn.functional as F
import torch.nn as nn

class YOLOWithDifferentiableNMS(nn.Module):
    def __init__(self, yolo_model):
        super(YOLOWithDifferentiableNMS, self).__init__()
        self.yolo_model = yolo_model

    def forward(self, imgs):
        # Forward pass through YOLO
        output = self.yolo_model(imgs)
        boxes, scores = self.extract_boxes_and_scores(output)
        # Apply softmax-weighted averaging to get the final bounding box
        final_box = self.soft_nms_single_box(boxes, scores)
        return final_box

    def extract_boxes_and_scores(self, yolo_output):
        # Extract bounding boxes and scores from YOLO output
        # This function will be specific to the YOLO model's output format
        boxes = yolo_output[:, :, :4]  # assuming the first 4 values are the box coordinates
        scores = yolo_output[:, :, 4] *  yolo_output[:, :, 5] # multiply obj score by person confidence
        return boxes, scores


    # soft nms decays the confidence scores of bounding boxes based on their overlap with the highest scoring box
    def soft_nms_single_box(self, boxes, scores, iou_threshold=0.5):
        # Compute IoU matrix
        iou_matrix = self.compute_iou_matrix(boxes)
        # Apply softmax to scores adjusted by IoU
        adjusted_scores = scores.clone()
        for i in range(len(boxes)):
            overlaps = iou_matrix[i]
            weight = torch.exp(-overlaps / iou_threshold)
            adjusted_scores[i] = (weight * scores).sum()
        # Apply softmax to get weighting factors
        weights = F.softmax(adjusted_scores, dim=0)
        # Compute the weighted average bounding box
        weighted_box = (weights[:, None] * boxes).sum(dim=0)
        return weighted_box

    def compute_iou_matrix(self, boxes):
        N = boxes.shape[0]
        iou_matrix = torch.zeros((N, N))
        for i in range(N):
            for j in range(N):
                iou_matrix[i, j] = self.compute_iou(boxes[i], boxes[j])
        return iou_matrix

    def compute_iou(self, box1, box2):
        inter_xmin = max(box1[0], box2[0])
        inter_ymin = max(box1[1], box2[1])
        inter_xmax = min(box1[2], box2[2])
        inter_ymax = min(box1[3], box2[3])
        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area
        return iou

# Example usage
class MockYOLOModel(nn.Module):
    def forward(self, x):
        # Mock output for demonstration: (batch_size, num_boxes, 5) -> (x1, y1, x2, y2, score)
        return torch.randn(x.size(0), 10, 5)

yolo_model = MockYOLOModel()
model = YOLOWithDifferentiableNMS(yolo_model)

# Forward pass with a mock image batch
imgs = torch.randn(2, 3, 256, 256)  # batch of 2 images, 3 channels, 256x256
output_boxes = model(imgs)
print(output_boxes)

# Loss computation and backpropagation
loss = compute_loss(output_boxes)  # Define your custom loss function
loss.backward()


# def soft_nms_single_box(boxes, scores, iou_threshold=0.5, score_threshold=0.001):
#     # Compute IoU matrix
#     iou_matrix = compute_iou_matrix(boxes)
    
#     # Apply softmax to scores adjusted by IoU
#     adjusted_scores = scores.clone()
#     for i in range(len(boxes)):
#         overlaps = iou_matrix[i]
#         weight = torch.exp(-overlaps / iou_threshold)
#         adjusted_scores[i] = (weight * scores).sum()
    
#     # Apply softmax to get weighting factors
#     weights = F.softmax(adjusted_scores, dim=0)
    
#     # Compute the weighted average bounding box
#     weighted_box = (weights[:, None] * boxes).sum(dim=0)
    
#     return weighted_box

# def compute_iou_matrix(boxes):
#     N = boxes.shape[0]
#     iou_matrix = torch.zeros((N, N))
#     for i in range(N):
#         for j in range(N):
#             iou_matrix[i, j] = compute_iou(boxes[i], boxes[j])
#     return iou_matrix

# def compute_iou(box1, box2):
#     inter_xmin = max(box1[0], box2[0])
#     inter_ymin = max(box1[1], box2[1])
#     inter_xmax = min(box1[2], box2[2])
#     inter_ymax = min(box1[3], box2[3])
#     inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     union_area = box1_area + box2_area - inter_area
#     iou = inter_area / union_area
#     return iou

# # Example usage
# boxes = torch.tensor([
#     [10.5, 20.5, 30.5, 40.5],
#     [12.0, 22.0, 32.0, 42.0],
#     [11.0, 21.0, 31.0, 41.0]
# ])
# scores = torch.tensor([0.9, 0.8, 0.85])

# single_box = soft_nms_single_box(boxes, scores)
# print(single_box)
