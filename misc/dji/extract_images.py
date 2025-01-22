import cv2 
import os 
from tqdm import trange
  
# Read the video from specified path 
camera = cv2.VideoCapture('videos/dji_fly_20250121_135614_0011_1737464791745_video.mp4')
ret, img = camera.read()

camera.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
num_frames = camera.get(cv2.CAP_PROP_FRAME_COUNT)
camera.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
  
os.makedirs('images', exist_ok=True)
  
# check if images are already in the folder
img_counter = len(os.listdir('images'))

for current_frame in trange(int(num_frames)):
    ret, frame = camera.read()
    if ret: 
        # Saves the frames
        name = f'images/{current_frame+img_counter:04d}.jpg'
        cv2.imwrite(name, frame) 
  
    else: 
        break
  
# Release all space and windows once done 
camera.release() 
cv2.destroyAllWindows() 
