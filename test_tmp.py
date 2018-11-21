import cv2

img = cv2.imread("test_imgs/undist_test.jpg")

box = cv2.selectROI("Frame", img, fromCenter=False,
            showCrosshair=True)

print(box)
