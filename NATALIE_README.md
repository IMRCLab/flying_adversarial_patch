## YOLO Adversarial Patches
### src/attacks.py
`python3 attacks.py --file settings.yaml --task task_num`
- added task_num param (int) for generating many patches in parallel
- changed targeted_attack_joint to use the YOLOBounding class to predict poses instead
- currently, only targets that are valid in the image pixel context are generated

### src/run_attacks.py
- generates many patches in parallel using multiprocessing module
- change `start_task_id`, `end_task_id`, and `num_runs`
- change `OUTPUT_PATH` to save log files (stderr and stdout)

### src/test_yolo_patches.py 
`python3 src/test_yolo_patches.py --start start_patch_id --end end_patch_id`
- functions for calculating error of yolo patches and displaying them (params are starting and ending patch_id)
- expects that patches are saved in PATH folder in the format that `attacks.py` saves them in (npy files), and each patch has a corresponding `settings_patchid.yaml` file
<br><br/>
For each patch:
1. loads the patch from `PATH/last_patch_id.npy`
2. loads the optimized position from `PATH/position_norm_id.npy`
3. loads the targets from `PATH/settings.yaml`
4. for each target, generates an image from the dataset with the patch placed at the optimized position for that target
5. for each patch, calculates the MSE between the center of the highest confidence bounding box and the actual target for a random image in the dataset for all targets
6. for each patch/target pair, saves an image in `RESULT_PATH/yolo_boxes_{patch_id}_target_{target_id}.png` that displays the patch placed on the image, the target, and the predicted YOLO bounding boxes (red is highest confidence, then blue, green, purple) <br> See example in `test_yolo_example.png`

The error printed isn't the error over the entire dataset for all patches, just the average loss for all patches when placed on ONE randomly selected image from the Frontnet dataset.

### src/yolo_bounding.py
has class `YOLOBox`, a wrapper around the YOLO ultralytics model that skips the normal NMS step and takes a weighted average to calculate the highest confidence bounding box so that the output is differentiable
- uses the ultralytics YOLOv5 nano module with `autoshape=False` so that it can take batched inputs
- the score assigned to each box is the product of the objectness score and the person_class score, calculated in `extract_boxes_and_scores`
- used in `attacks.py` to train patches against the YOLO model

### src/camera.py
- wrapper class around camera functions, takes a cam config file as param
Functions: 
- `xyz_from_bb` calculates xyz coords from a bounding box
- `tensor_xyz_from_bb` same thing, but with tensors
- `point_from_xyz` calculates image point from xyz coords

### src/create_dataset.py
- creates a pkl file that stores patch data
- call `save_pickle(start_patch_id, end_patch_id)` to create a pkl file for patches from start_id to end_id inclusive
- saves only the best target for each patch

### Other
All the patches I generated (653, but only 578 with valid targets) are saved in `all_yolo_patches.pkl`
Diffusion model trained for 1000 epochs, 1000 denoising steps (approx 23 min training) is saved in `yolo_conditioned_unet_80x80_1000_3256i_255.pth ` <br>
Samples from that model saved in `samples_conditioning_4_80x80.png`