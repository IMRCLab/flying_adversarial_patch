import pandas as pd
import numpy as np
import rowan
import cv2
import torch


SOFTMAX_MULT = 8.
RADIUS = 0.3 # in m

# scale box to original image scale
def scale_box(box, scale_factor_width, scale_factor_height):
    box[0] /= scale_factor_width
    box[1] /= scale_factor_height
    box[2] /= scale_factor_width
    box[3] /= scale_factor_height
    return box

# compute relative position of center of patch in camera frame
def xyz_from_bb(bb, fx, fy, ox, oy, camera_extrinsic):
    
    # get pixels for bb side center
    P1 = np.array([bb[0],(bb[1] + bb[3])/2])
    P2 = np.array([bb[2],(bb[1] + bb[3])/2])

    # print(P1, P2)

    # get rays for pixels
    a1 = np.array([(P1[0]-ox)/fx, (P1[1]-oy)/fy, 1.0])
    a2 = np.array([(P2[0]-ox)/fx, (P2[1]-oy)/fy, 1.0])

    # normalize rays
    a1_norm = np.linalg.norm(a1)
    a2_norm = np.linalg.norm(a2)

    # get the distance    
    distance = (np.sqrt(2)*RADIUS)/(np.sqrt(1-np.dot(a1,a2)/(a1_norm*a2_norm)))

    ac = (a1+a2)/2

    # get the position
    xyz = distance*ac/np.linalg.norm(ac)
    # print((np.linalg.inv(camera_extrinsic) @ [*xyz, 1]))
    # print((np.linalg.inv(camera_extrinsic) @ [*xyz, 1]).transpose())
    new_xyz = (np.linalg.inv(camera_extrinsic) @ [*xyz, 1])[:3]
    return new_xyz

# taken from ultralytics yolo
def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def load_feather(sensor_name, path):
    df_intrinsic = pd.read_feather(path + '/intrinsics.feather')
    df_extrinsic = pd.read_feather(path + '/egovehicle_SE3_sensor.feather')

    data_intrinsic = df_intrinsic[df_intrinsic['sensor_name'] == sensor_name].to_dict('records')[0]
    data_extrinsic = df_extrinsic[df_extrinsic['sensor_name'] == sensor_name].to_dict('records')[0]

    return data_intrinsic, data_extrinsic


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load image from path with cv2
img = cv2.imread('misc/argoverse/0/ring_side_right/315967378977482497.jpg')
cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# scale image to height = 640
# print(img.shape)
scale_factor_height = 320 / img.shape[0]
scale_factor_width = 640 / img.shape[1]
img_r = cv2.resize(img, (640, 320))
# print(img.shape)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
img_t = torch.tensor(img_r).permute(2, 0, 1).unsqueeze(0).float() / 255.
img_t = img_t.requires_grad_(True).to(device)
print(img_t.shape)
print(img_t.min(), img_t.max())
# load yolov5
model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True, autoshape=False)
model.eval()
# get camera calibration

def get_camera_intrinsic_matrix(sensor_name, data):
    # see https://github.com/argoverse/argoverse-api/blob/master/argoverse/utils/calibration.py#L282

    matrix = np.zeros((3, 3))
    matrix[0, 0] = data['fx_px']
    matrix[0, 2] = data['cx_px']
    matrix[1, 1] = data['fy_px']
    matrix[1, 2] = data['cy_px']
    matrix[2, 2] = 1.0

    distortion_coeffs = np.array([data['k1'], data['k2'], data['k3']])

    img_width = data['width_px']
    img_height = data['height_px']

    return matrix, distortion_coeffs, (img_width, img_height)

def quat2rot(q):
    norm = np.linalg.norm(q)
    if not np.isclose(norm, 1.0, atol=1e-12):
        q = q / norm
    
    matrix = rowan.to_matrix(q)
    return matrix


def get_camera_extrinsic_matrix(sensor_name, data):
    # see https://github.com/argoverse/argoverse-api/blob/master/argoverse/utils/calibration.py#L258
    egovehicle_t_camera = np.array([data['tx_m'], data['ty_m'], data['tz_m']])
    egovehicle_q_camera = np.array([data['qw'], data['qx'], data['qy'], data['qz']])

    egovehicle_R_camera = quat2rot(egovehicle_q_camera)

    # inverse of egovehicle_T_camera
    # see https://github.com/argoverse/argoverse-api/blob/master/argoverse/utils/se3.py#L42
    camera_T_world = np.zeros((4,4))
    camera_T_world[:3, :3] = egovehicle_R_camera.T
    camera_T_world[:3, 3] = np.dot(egovehicle_R_camera.T, -egovehicle_t_camera)
    camera_T_world[3, 3] = 1.0

    return camera_T_world
    
def get_camera_calibration(sensor_name, path):
    data_intrinsic, data_extrinsic = load_feather(sensor_name, path)
    intrinsic, distortion, img_dim = get_camera_intrinsic_matrix(sensor_name, data_intrinsic)
    extrinsic = get_camera_extrinsic_matrix(sensor_name, data_extrinsic)

    return intrinsic, extrinsic, distortion, img_dim




intrinsic, extrinsic, distortion, img_dim = get_camera_calibration('ring_side_right', 'misc/argoverse/0/calibration')
print(intrinsic)
print(extrinsic)
print(distortion)



fx = intrinsic[0, 0]
fy = intrinsic[1, 1]
ox = intrinsic[0, 2]
oy = intrinsic[1, 2]

# predict boxes
results = model(img_t)[0]

print(results.shape) # shape (1, num candidate boxes, 80 class scores + 4 box locations (x, y, width, height) + 1 objectness score)
print(results.grad_fn)

# select only boxes with highest score in class person

boxes = xywh2xyxy(results[:, :, :4])
# print(boxes.grad_fn)
scores = results[:, :, 4] * results[0][:, 5] # multiply obj score by person confidence

print(boxes.shape)
print(scores.shape)

max_score_idx = torch.argmax(scores)

# print(max_score_idx)
best_box = boxes[0, max_score_idx].detach().cpu().numpy()
# best_box = xywh2xyxy(best_box)
print(best_box.shape)

# scale those bounding box back to original image scale
best_box = scale_box(best_box, scale_factor_width, scale_factor_height)

print(best_box)

xyz = xyz_from_bb(best_box, fx, fy, ox, oy, extrinsic)
print('Person xyz: ', xyz)


# calculate center of best_box
center_x = (best_box[0] + best_box[2]) / 2
center_y = (best_box[1] + best_box[3]) / 2

#sanity check 3d coordinates in world to 2d in image
xyz_hom = np.append(xyz, 1.0)
print(xyz_hom)
uv_cam = np.dot(extrinsic, xyz_hom.transpose())[:3]
print(uv_cam)
uv = np.dot(intrinsic, uv_cam)
uv = (uv / uv[2])[:2]
print("Calc img coords center: ", uv)

print("GT img coords center: ", (center_x, center_y))

# Sanity check passed! :D

# # project center to 3D
# center_3d = np.array([center_x, center_y, 1.0])
# center_3d = np.dot(np.linalg.inv(intrinsic), center_3d)
# center_3d = center_3d / center_3d[2]
# print(center_3d)

# # project 3D point to 3D world
# center_3d = np.append(center_3d, 1.0)
# center_3d = np.dot(np.linalg.inv(extrinsic), center_3d)
# center_3d = center_3d / center_3d[3]
# print(center_3d)



# take a weighted average of the boxes
soft_scores = torch.nn.functional.softmax(scores * SOFTMAX_MULT)
print(soft_scores.shape)

# plot scores and soft scores in two separate sub figures
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2)
axs[0].plot(scores[0].detach().cpu().numpy())
axs[1].plot(soft_scores[0].detach().cpu().numpy())
plt.savefig('scores.jpg')

print("max soft scores: ", torch.argmax(soft_scores))

# weighted_boxes = torch.sum(boxes * soft_scores.unsqueeze(2), dim=1) / torch.sum(scores)
# print(weighted_boxes.shape)
# print(weighted_boxes)
# print(weighted_boxes.grad_fn)

selected_box = torch.bmm(soft_scores.unsqueeze(1), boxes).squeeze(1).detach().cpu().numpy()
print(selected_box.shape) 
print(selected_box)


selected_box = scale_box(selected_box[0], scale_factor_width, scale_factor_height).reshape(1, -1)

print(selected_box)

#soft_scores = torch.nn.functional.softmax(scores * SOFTMAX_MULT, dim=0)


# weighted_boxes = torch.sum(boxes * soft_scores.unsqueeze(-1), dim=0) / torch.sum(scores)
# print(weighted_boxes.shape)

xmin, ymin, xmax, ymax = best_box
cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 10)
cv2.drawMarker(img, (int(center_x), int(center_y)), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=10)

xmin, ymin, xmax, ymax = selected_box[0]
cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 10)

cv2.imwrite('output.jpg', img)



# print(results[0])
# print(results[0])


# boxes = xywh2xyxy(results[0][:, :4])
# print(boxes.shape)
# scores = results[0][:, 4] *  results[0][:, 5] # multiply obj score by person confidence

# print(scores.shape)

# # take a weighted average of the boxes
# soft_scores = torch.nn.functional.softmax(scores * SOFTMAX_MULT, dim=-1)
# soft_scores = soft_scores.unsqueeze(1)
# print(soft_scores.shape)
# selected_boxes = torch.bmm(soft_scores, boxes)

# print(selected_boxes.shape, selected_boxes)



# print(boxes.shape)
# # print(boxes)
# xmin, ymin, xmax, ymax = boxes[0]
# print(xmin, ymin, xmax, ymax)
# cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
# cv2.imwrite('output.jpg', img)


# intrinsic, distortion, img_dim = get_camera_intrinsic_matrix('ring_side_right', df)
# print(intrinsic)
# print(distortion)
# print(img_dim)