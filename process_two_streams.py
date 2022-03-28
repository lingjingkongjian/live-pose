import cv2
import os
from dlclive import DLCLive, Processor
from aniposelib.cameras import Camera, CameraGroup
import numpy as np


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
dlc_proc = Processor()
dlc_live = DLCLive("DLC_human_dancing_resnet_101_iteration-0_shuffle-1", processor=dlc_proc)

# v4l2-ctl --list-devices

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)
cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # set buffer size
cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # set buffer size

ret1, img1 = cap1.read()
ret2, img2 = cap2.read()

dlc_live.init_inference(img1)
cgroup = CameraGroup.load('calibration.toml')

while 1:

  ret1, img1 = cap1.read()
  ret2, img2 = cap2.read()

  if ret1 and ret2:
      poses1 = dlc_live.get_pose(img1)
      print(poses1)
      poses2 = dlc_live.get_pose(img2)
      print(poses2)
      points = np.concatenate((poses1, poses2), axis=0)
      print(points)

      #score_threshold = 0.5
      # remove points that are below threshold
      #points[scores < score_threshold] = np.nan

      points_flat = points.reshape(2, -1, 2)
      #scores_flat = scores.reshape(n_cams, -1)

      p3ds_flat = cgroup.triangulate(points_flat, progress=True)
      reprojerr_flat = cgroup.reprojection_error(p3ds_flat, points_flat, mean=True)
      print(reprojerr_flat)

      cv2.imshow('img1',img1)
      cv2.imshow('img2',img2)

      k = cv2.waitKey(100) 
      if k == 27: #press Esc to exit
         break

cap1.release()
cap2.release()
cv2.destroyAllWindows()

## example triangulation without filtering, should take < 15 seconds
fname_dict = {
    'A': '2019-08-02-vid01-camA.h5',
    'B': '2019-08-02-vid01-camB.h5',
    'C': '2019-08-02-vid01-camC.h5',
}

