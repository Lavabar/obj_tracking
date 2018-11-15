from cv2 import createBackgroundSubtractorMOG2 as createMOG2
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fgbg = createMOG2()

ret, bg = cap.read()
#n = 1
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
#bg = np.zeros((480, 640))
while True:
#    n += 1
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    bgmask = np.invert(fgmask)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for y in range(480):
        for x in range(640):
            if bgmask[y, x] != 255:
                frame[y, x] = 0
    
    #bg = np.subtract(bg, fgmask)
    #bg = np.add(bg, bgmask)
#    bg = bg + frame
#    bg = np.true_divide(bg, n)
    #bg = np.logical_or(bg, bgmask)

    cv2.imshow("frame", frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
