import cv2
import os
from aniposelib.cameras import Camera, CameraGroup
import numpy as np

method = "voxelpose"
if(method == "dlc"):
    from dlc import setup
    from dlc import inference
elif(method == "voxelpose"):
    from voxelpose import setup
    from voxelpose import inference

# v4l2-ctl --list-devices

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)
cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # set buffer size
cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # set buffer size

ret1, img1 = cap1.read()
ret2, img2 = cap2.read()

setup(img1) # some methods need the first frame to initialise
cgroup = CameraGroup.load('calibration.toml')

while 1:

  ret1, img1 = cap1.read()
  ret2, img2 = cap2.read()

  if ret1 and ret2:
      
      points2D, points3D = inference([img1, img2], cgroup)
      reprojerr_flat = cgroup.reprojection_error(points3D, points2D, mean=True)
      #print(reprojerr_flat)

      cv2.imshow('img1',img1)
      cv2.imshow('img2',img2)

      k = cv2.waitKey(100) 
      if k == 27: #press Esc to exit
         break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
