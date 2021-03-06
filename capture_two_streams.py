import cv2
# v4l2-ctl --list-devices

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

while 1:

  ret1, img1 = cap1.read()
  ret2, img2 = cap2.read()

  if ret1 and ret2:

      cv2.imshow('img1',img1)
      cv2.imshow('img2',img2)

      k = cv2.waitKey(100) 
      if k == 27: #press Esc to exit
         break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
