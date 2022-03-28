# live-pose
Live pose estimation with SOTA algorithms from webcams / (later) Autopilot

## Single-view pose estimation
- https://github.com/freemocap/freemocap

## Single-person multi-view pose estimation
- https://github.com/microsoft/multiview-human-pose-estimation-pytorch
- https://github.com/karfly/learnable-triangulation-pytorch

## Multi-person multi-view pose estimation
- https://github.com/microsoft/voxelpose-pytorch
- https://github.com/zhangyux15/4d_association
- https://github.com/Jeff-sjtu/CrowdPose
- https://github.com/zju3dv/mvpose
- https://github.com/lambdaloop/anipose

# use conda according to 
https://github.com/DeepLabCut/DeepLabCut-live/blob/master/docs/install_desktop.md
for test use export TF_FORCE_GPU_ALLOW_GROWTH=true

# install tensorrt
os="ubuntu2004"
tag="cudax.x-trt8.x.x.x-yyyymmdd"
sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-${os}-${tag}/7fa2af80.pub

sudo apt-get update
sudo apt-get install tensorrt