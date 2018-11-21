import cv2
import numpy as np
from math import atan
import time
def undistortion(img, strength=3, zoom=1)#, refPt):
    dest_img = np.zeros(img.shape, dtype=np.uint8)

    imageHeight, imageWidth = img.shape
    
    halfWidth = imageWidth / 2
    halfHeight = imageHeight / 2
    
    correctionRadius = (imageWidth**2 + imageHeight**2)**0.5 / strength

    for y in range(imageHeight):
        for x in range(imageWidth):
            newX = x - halfWidth
            newY = y - halfHeight

            distance = (newX**2 + newY**2)**0.5
            r = distance / correctionRadius
            
            theta = 0.0
            if r == 0:
                theta = 1
            else:
                theta = atan(r) / r

            sourceX = int(halfWidth + theta * newX * zoom)
            sourceY = int(halfHeight + theta * newY * zoom)
            
            dest_img[y, x] = img[sourceY, sourceX]

            
    return dest_img

if __name__ == "__main__":
    img = cv2.imread("test_imgs/undist_test2.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    start = time.time()
    undist = undistortion(img)
    print(time.time() - start)
    cv2.imshow("undistortioned", undist)

    key = False
    while not key:
        key = cv2.waitKey(1) & 0xFF == ord('q')
    cv2.destroyAllWindows()