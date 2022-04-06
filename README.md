# live-pose
Live pose estimation with SOTA algorithms from webcams

# advanced camera setup
- Autopilot (later)
- https://www.dev47apps.com/
- https://reincubate.com/camo/

# Why this?
Datasets are usually captured in lab conditions. We want to provide an easy way to evaluate and compare the generalisation ability of SOTA algorithms on live real-world data. In addition, we provide an interface for people to use their own algorithms for the Lingjing Tech Poseteacher VR application (coming soon!) . 

# how to run
python process_two_streams.py

# TODO:
- buy more cameras / more camera streams
- integrate with raspberry pi (Autopilot)
- mmpose
- visualisation of 2D detections
- visualisation of 3D detections
- add code for Websocket connection to Poseteacher

## Single-view pose estimation
- https://github.com/freemocap/freemocap (single person)
- https://github.com/mks0601/3DMPPE_POSENET_RELEASE (multi person)

## Single-person multi-view pose estimation
- https://github.com/microsoft/multiview-human-pose-estimation-pytorch
- https://github.com/karfly/learnable-triangulation-pytorch
- https://github.com/HowieMa/TransFusion-Pose

## Multi-person multi-view pose estimation
- https://github.com/microsoft/voxelpose-pytorch
- https://github.com/zhangyux15/4d_association
- https://github.com/Jeff-sjtu/CrowdPose
- https://github.com/zju3dv/mvpose
- https://github.com/lambdaloop/anipose

# Installation
## with Deeplabcut
use conda according to 
https://github.com/DeepLabCut/DeepLabCut-live/blob/master/docs/install_desktop.md
for test use export TF_FORCE_GPU_ALLOW_GROWTH=true
conda activate dlc-live
## with MMPose
https://github.com/open-mmlab/mmpose/blob/master/docs/en/install.md 

# install tensorrt
os="ubuntu2004"
tag="cudax.x-trt8.x.x.x-yyyymmdd"
sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-${os}-${tag}/7fa2af80.pub

sudo apt-get update
sudo apt-get install tensorrt


# errors
## calibration
ValueError: not enough values to unpack (expected 2, got 0) --> make sure corners of Checkerboard are in the view

## MMPpose
MMPose errors: from mmdet.apis import inference_detector, init_detector --> https://github.com/open-mmlab/mmpose/issues/530 Please uninstall all mmcv & mmcv-full in your system, and reinstall the latest mmcv-full.

https://github.com/facebookresearch/detectron2/issues/686
