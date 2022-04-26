from dlclive import DLCLive, Processor
import numpy as np
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
dlc_live = None

def setup(img1):
    global dlc_live
    dlc_proc = Processor()
    dlc_human_path = "DLC_human_dancing_resnet_101_iteration-0_shuffle-1"

    dlc_live = DLCLive(dlc_human_path, processor=dlc_proc)
    dlc_live.init_inference(img1)


def inference(imgs, cgroup):
    img1 = imgs[0]
    img2 = imgs[1]

    poses1 = dlc_live.get_pose(img1)
    #print(poses1)
    poses2 = dlc_live.get_pose(img2)
    #print(poses2)
    points = np.concatenate((poses1, poses2), axis=0)

    points_flat = points.reshape(2, -1, 2)

    p3ds_flat = cgroup.triangulate(points_flat, progress=True)

    return points_flat, p3ds_flat
