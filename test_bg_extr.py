import cv2
import numpy as np
from scipy.ndimage import median_filter

N_FRAMES = 100

global stats, n
stats = np.zeros((480, 640, 256))
n = 0

def update_bg(img, bg):

    global stats, n

    # TODO - optimization
    for y in range(480):
        for x in range(640):
            stats[y, x, img[y, x]] += 1
    
    n += 1
    
    # alternative variant (slower)
    #img = to_categorical(img, num_classes=256)
    #stats = np.add(stats, img)
    
    if n == N_FRAMES:
        bg = np.argmax(stats, axis=2)
        stats = np.zeros((480, 640, 256))
        n = 0
        
    return bg


cap = cv2.VideoCapture(0)

if __name__ == "__main__":
    ret, bg = cap.read()
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        bg = update_bg(gray, bg)

        cv2.imshow("bg", median_filter(np.array(bg, np.uint8), 3))
        cv2.imshow("frame", gray)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
