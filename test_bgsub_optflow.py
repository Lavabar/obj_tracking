import cv2
import numpy as np
import os

path = "test_videos/Crowd_PETS09/S1/L1/Time_13-59/View_002/"
ls = os.listdir(path)
ls.sort()

frame = cv2.imread(path+ls[0])
prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame)
hsv[...,1 ] = 255

for i in range(1, len(ls)):
    img = cv2.imread(path+ls[i])
    nxt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    cv2.imshow('frame2',bgr)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',img)
        cv2.imwrite('opticalhsv.png',bgr)
    prvs = nxt
cv2.destroyAllWindows()