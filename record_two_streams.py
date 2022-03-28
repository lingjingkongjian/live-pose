import cv2
# v4l2-ctl --list-devices

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)
# attention: this depends on the camera
cap1.set(cv2.CAP_PROP_FPS, 30)
cap2.set(cv2.CAP_PROP_FPS, 30)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out1 = cv2.VideoWriter('output1.mp4', fourcc, 10.0, (640,480))
out2 = cv2.VideoWriter('output2.mp4', fourcc, 10.0, (640,480))

while 1:

  ret1, img1 = cap1.read()
  ret2, img2 = cap2.read()

  if ret1 and ret2:

      cv2.imshow('img1',img1)
      cv2.imshow('img2',img2)
      out1.write(img1)
      out2.write(img2)

      k = cv2.waitKey(100) 
      if k == 27: #press Esc to exit
        print("received escape.")
        break

cap1.release()
cap2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()
