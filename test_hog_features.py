import cv2
import numpy as np
import imutils

if __name__ == "__main__":
    img_path = "test_imgs/frame_0214.jpg"
    img = cv2.imread(img_path)
    
    box = cv2.selectROI("Frame", img, fromCenter=False,
            showCrosshair=True)
    x, y, w, h = box

    roi_img = img[y:y+h, x:x+w]
    roi_img = imutils.resize(roi_img, width=64, height=128)

    norm_roi = np.float32(roi_img) / 255.0
    gx = cv2.Sobel(norm_roi, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(norm_roi, cv2.CV_32F, 0, 1, ksize=1)

    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    mag *= 255.0
    mag = np.uint8(mag)
    cv2.imshow('frame', mag)
    while True:
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break