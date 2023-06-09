# Flying Adversarial Patches for DL-based multirotors

## Installation
### Clone the repository
To clone the repository including the code from the pulp-frontnet module, use this command:
```bash
# via ssh
$ git clone --recurse-submodules git@github.com:phanfeld/flying_adversarial_patch.git
# or via https
$ git clone --recurse-submodules https://github.com/phanfeld/flying_adversarial_patch.git
```

If you have cloned the repository without the `--recurse-submodules` argument and want to pull the pulp-frontnet code, please use the following command inside the repository:
```bash
$ git submodule update --init --recursive
```
### Download the datasets
For downloading the datasets:
```bash
$ cd pulp-frontnet/PyTorch
$ curl https://drive.switch.ch/index.php/s/FMQOLsBlbLmZWxm/download -o pulp-frontnet-data.zip
$ unzip pulp-frontnet-data.zip
$ rm pulp-frontnet-data.zip
```
The datasets should now be located at `pulp-frontnet/PyTorch/Data/`.

### Setting up a Python Virtual Environment and installing needed Python packages
To install all of the needed Python packages, you can use the provided `requirements.txt` file. To avoid messing with packages you need for other projects, you can set up a Python Virtual Environment. To do so, execute the following commands.
```bash
$ python3 -m venv /path/to/env
$ source path/to/env/bin/activate
$ python -m pip install -r path/to/flying_adversarial_patch/requirements.txt
```


## Compute adversarial patch and position
To generate the adversarial patch with optimal transformation matrices, you can call
```bash
$ python src/attacks.py --file settings.yaml
```
Please adapt the hyperparameters in `settings.yaml` in the main folder according to your needs.
### Choosing the optimization approach
Please change the optimization approach in the `settings.yaml` file to your desired mode. You can choose between `'fixed'`, `'joint'`, `'split'`, and `'hybrid'`.
```yaml
mode: 'split' # 'fixed', 'joint', 'split', 'hybrid'
```
### Setting target positions
For setting multiple desired target positions $\bar{\mathbf{p}}^h_K$, change the values in `settings.yaml` for the targets like so:
```yaml
  x : [1.0, 0.5]
  y : [-1, 1]
  z : [0, null]
```
Now, the patch will be optimized for two targets: $\bar{\mathbf{p}}^h_1 = (1, -1, 0)^T$, $\bar{\mathbf{p}}^h_2 = (0.5, 1, z)^T$. For the second target, the attack does not set the $z$-value to a desired one but tries to keep it to the originally predicted $z$ in $\hat{\mathbf{p}}^h$ for the current image.
### Starting from different initial patch
You can change the initial patch for a training run in the settings file. Either set
```yaml
patch: 
  mode: 'face'
  path: 'src/custom_patches/custom_patch_resized.npy'
```
to, e.g., start from a patch showing a face. Please specify the path to point to a valid numpy array file. The patch should be grayscale and can have any width and height. We prepared multiple patches in the `src/custom_patches` folder.

If the initial patch should be white or starting from random pixel values, adapt the patch mode in the `settings.yaml` like:
```yaml
patch: 
  mode: 'white'
```
or 
```yaml
patch: 
  mode: 'random'
```
### Results
All results will be saved at the specified path in the `settings.yaml`.\
The folder will contain the following files:
```
path
|_settings.yaml # a copy of the settings.yaml
|_patches.npy   # a numpy array containing all patches
|_positions_norm.npy # the optimized positions for the K targets
|_positions_losses.npy # all computed losses for the positions
|_patch_losses.npy # all computed losses for the current patch
|_losses_test.npy # the loss on the entire testset after each iteration
|_losses_test.npy # the loss for the entire trainset after each iteration
|_boxplot_data.npy # an array containing all of the data needed to create the boxplots from the paper
```
## Reproduce the experiments of the paper
To reproduce all of the results from the paper "Flying Adversarial Patches: Manipulating the Behavior of Deep Learning-based Autonomous Multirotors", we prepared several scripts:
### Comparison between the different approaches
To run the full experiment on the different approaches, run:
```bash
python src/exp1.py --file exp1.yaml -j 4 --trials 10 --mode all
```

Please adapt the hyperparameters in the `exp1.yaml` file according to your needs. 

With `-j 4`, 4 worker processes are spawned and all approaches are computed in parallel. Depending on your hardware, you can set `-j` to a different value. If `-j` is set to 1, the different approaches will be computed consecutively.

With `--trials 10` you can set the number of paraellel training runs for the same mode to 10 like we did in the paper.

With `--mode all` you can choose all modes ('fixed', 'joint', 'split', 'hybrid'). You can additionally set the mode to one or a combination of all modes with, e.g., `--mode fixed hybrid` to only run the experiment for the 'fixed' and 'hybrid' approach.

The resulting mean test loss for all optimization approaches will be printed in the terminal.\
The results folder will contain a PDF file including the boxplots (among others) similar to Fig. 3 and 4 from the paper.

### Scalability for multiple target positions
To run the experiment on $1\leq K \leq 10$ desired target positions $\bar{\mathbf{p}}^h_K$, run:
```bash
python src/exp2.py --file exp2.yaml -j 4 --trials 10
```

Please adapt `exp2.yaml` according to your needs. Note that the mode needs to be changed in the yaml file! Setting the mode with the `--mode` argument is not possible (currently).

The resulting mean test loss for all $K$ will be printed in the terminal.\
The results folder will contain a PDF file including a plot similar to Fig. 5 from the paper.

### Comparison different starting patches
You can reproduce the experiment analyzing different starting patches with executing:
```bash
python src/exp3.py --file exp3.yaml -j 4 --trials 10 --mode all
```

Please adapt `exp3.yaml` according to your needs.

The resulting mean test loss for all patch modes and optimization approaches will be printed in the terminal.


## Reproduce camera calibration
The camera calibration was performed on the `160x96StrangersTestset` dataset provided by the pulp-frontnet authors. If you followed the steps in [Download the datasets](#download-the-datasets), you can find the dataset here: `pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle`.

We saved all the 3D coordinates and the corresponding 2D coordinates of the humans in the images in a csv file. You can find it here: `adversarial_frontnet/camera_calibration/ground_truth_pose.cs`

The 3D coordinates are relative to the camera - the UAV with an attached [AI deck](https://www.bitcraze.io/documentation/tutorials/getting-started-with-aideck/). These values are stored as ground-truth data in the dataset.

The 2D coordinates of the human in the image are manually annotated and therefore prone to errors.

We investigated two ways to ways to calibrate the camera: 
1) calculating a projection matrix with *Direct Linear Transformation* 
2) utilizing OpenCV's `calibrateCamera()` and `projectPoints()` functions

We have calculated the l2 distance between the manually set points stored in the csv file and the calculated pixel coordinates utilizing both methods. The mean l2 distance of the calculated pixel coordinates utilizing OpenCV was smaller. We therefore adapted the OpenCV functions for our code.

You can follow the main method in `adversarial_frontnet/camera_calibration/camera_calibration.py` to calculate the camera intrinsics, rotation and translation matrix and the distortion coefficients needed for projecting pixels. Additionally, you can load these values from the yaml file, provided in the same folder.

## Hardware
### Prerequisites:
* [Crazyflie 2.1](https://store.bitcraze.io/products/crazyflie-2-1)
* [Crazyradio PA 2.4 GHz](https://store.bitcraze.io/products/crazyradio-pa)
* [AI Deck 1.1](https://store.bitcraze.io/products/ai-deck-1-1?variant=32485907890263)
* JTAG Debugger with 10-pin-connector
* Gamepad/Controller (e.g. XBox 360 USB Controller)
* [Install Crazyflie Client](https://www.bitcraze.io/documentation/repository/crazyflie-clients-python/master/installation/install/)
* [Install docker](https://docs.docker.com/desktop/install/ubuntu/)
* [Install ros2 galactic](https://docs.ros.org/en/galactic/Installation.html)

For flying, the UAVs need to be aware of their current state estimates. This information is either provided through a motion capture system or the [Crazyflie FlowDeck v2](https://store.bitcraze.io/collections/decks/products/flow-deck-v2). State estimates provided by the FlowDeck are most likely more inaccurate than the pose information provided by a motion capture system.

### [Crazyswarm 2](https://imrclab.github.io/crazyswarm2/index.html)
For improved control of the UAVs, we'll utilize Crazyswarm 2.
#### Install dependencies:
```bash
$ sudo apt install libboost-program-options-dev libusb-1.0-0-dev
$ python -m pip install rowan 
# install python package rowan into your current python environment 
```
#### Set up your ros2 workspace:
Create a ros2 workspace. For ease of use, this folder can be placed in e.g. your home directory.
```bash
$ mkdir -p path/to/ros2_ws/src
$ cd path/to/ros2_ws/src
$ git clone https://github.com/IMRCLab/crazyswarm2 --recursive
$ git clone --branch ros2 --recursive https://github.com/IMRCLab/motion_capture_tracking.git
```
Additionally, the ros2 package provided with this repository needs to be accessible in the `ros2_ws/src` folder. Therefore, create a symbolic link:
```bash
$ ln -s path/to/adverserial_frontnet/hardware/frontnet_ros path/to/ros2_workspace/src
```
Please make sure to use the full path instead of relative paths to avoid issues with linking.\
Now build all of the packages:
```bash
$ cd ../ # go back to path/to/ros2_ws/
$ colcon build --symlink-install
```

### Crazyflie STM32 firmware
The Crazyflie firmware will be flashed wirelessly via the Crazyradio. Please power the Crazyflie via a battery or use the USB connector.\
As a prerequisite, we need the address of the Crazyflie. If not set manually, the standard address is `radio://0/80/2M/E7E7E7E7E7`. The address can be set easily with the [Crazyflie Client](https://www.bitcraze.io/documentation/repository/crazyflie-clients-python/master/userguides/userguide_client/#firmware-configuration).

Now move to the correct folder, build and lastly flash the firmware with the following commands:
```bash
$ cd path/to/adversarial_frontnet/hardware/frontnet_controller/
$ make -j
$ cfloader flash ../crazyflie-firmware/cf2.bin stm32-fw -w radio://0/80/2M/E7E7E7E7E7
```

### AI deck GAP8 Firmware (only needed for *victim* UAV)
To flash the quantized Frontnet network and adapted GAP8 firmware to the GAP8 of the AI deck, connect the JTAG Debugger to the corresponding pins of the GAP8 and via USB to your PC. Please attach the AI deck to the Crazyflie, such that it is powered either through the attached battery or the USB connector.

Then flash the code as follows:
```bash
$ cd path/to/adversarial_frontnet/hardware/frontnet_code/
$ docker run --rm -v ${PWD}:/module --device /dev/ttyUSB0 --privileged -P bitcraze/aideck /bin/bash -c 'export GAPY_OPENOCD_CABLE=interface/ftdi/olimex-arm-usb-tiny-h.cfg; source /gap_sdk/configs/ai_deck.sh; cd /module;  make clean all'
```
Please make sure that your JTAG device is `/dev/ttyUSB0`, otherwise please change the command accordingly with the correct number.

### Fly with Crazyswarm 2 and Frontnet
After successfull flashing of both firmwares to the STM32 and the GAP8, you can start your flight tests.\
If you are utilizing the FlowDeck for state estimates, make sure it is connected to the bottom of your Crazyflie. Otherwise make sure that your motion capture system is running and you have configured Crazyswarm 2 correctly to receive pose information (e.g. adapt `motion_capture.yaml` in the `hardware/frontnet_ros_config` folder to match your setup).

You'll need at least one terminal window opened in your ros2 workspace.
```bash
$ cd path/to/ros2_ws/
# additional sourcing needed to prepare ros2
$ source /opt/ros/galactic/setup.bash
$ . install/local_setup.bash
$ ros2 launch frontnet_ros launch.py
```
After a few seconds, you'll be able to take off pressing the Start button on the Xbox controller. To enable the Frontnet network output to be used to generate new setpoints, press the X button and move in front of the camera.

<!-- ### Generate C code and flashable image of quantized Frontnet
For creating a flashable image, we first need a `.onnx` file of the quantized networks. We use [nemo](https://github.com/pulp-platform/nemo) to receive the `.onnx` file.\
Nemo only works on older versions of PyTorch. We therefore create a new Python Virtual Environment for this process.
```bash
deactivate  # deactive your current virtual environment
python3 -m venv /path/to/nemo-env
source /path/to/nemo-env/bin/activate
python -m pip install torch==1.4.0 torchvision==0.5.0 pytorch-nemo==0.0.7 pandas==1.2.4 torchsummary==1.5.1 matplotlib==3.4.1
```

After installing nemo, please change directories back to the root directory of this repository.

You can now call the script, that the Frontnet authors provided for generating the `.onnx` file:
```bash
python pulp-frontnet/PyTorch/Scripts/QExport.py '160x32' --load-model 'pulp-frontnet/PyTorch/Models/Frontnet160x32.Q.pt' --load-trainset 'pulp-frontnet/PyTorch/Data/160x96OthersTrainsetAug.pickle' --regime pulp-frontnet/PyTorch/Scripts/regime.json
```

This will create a new folder `Results/160x32/Export` in which the `Frontnet.onnx` is saved.

For generating the flashable image, we'll utilize the Bitcraze AI deck docker image, since building depends partially on the GAP SDK.\
Run the docker image and mount this repository
```bash
docker run -it -v /path/to/adversarial_frontnet/:/home/adversarial_frontnet bitcraze/aideck
source /gap_sdk/configs/ai_deck.sh
cd /home/
```

We use DORY to generate a flashable image from the `.onnx` file.
```bash
# clone the repository
git clone https://github.com/pulp-platform/dory
cd dory
# we tested dory on this commit
git checkout 06b1b91fe1aa77f87b3baae97ee8dcb03eef1785
# get submodules
git submodule update --remote --init dory/Hardware_targets/GAP8/Backend_Kernels/pulp-nn
git submodule update --remote --init dory/Hardware_targets/GAP8/Backend_Kernels/pulp-nn-mixed
# install DORY as pip pickage into your current python environment
python -m pip install -e .
```

Now generate the image with the provided script:
```bash
python network_generate.py NEMO GAP8.GAP8_gvsoc /home/adversarial_frontnet/misc/dory_config.json --app_dir /home/adversarial_frontnet/hardware/frontnet_code/
```

TODO: Add section about adapting the source code before building!

Lastly, to generate the C code and flashable image:
```bash
cd /home/adversarial_frontnet/hardware/frontnet_code/
make clean all run CORE=8 platform=gvsoc
``` -->


