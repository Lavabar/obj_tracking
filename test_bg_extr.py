import cv2
import numpy as np

N_FRAMES = 10
#BACKGROUND = np.asarray([0, 200, 0])
#FOREGROUND = np.asarray([0, 200, 0])
FOREGROUND = 0

global stats, n
stats = np.zeros((480, 640, 256))
n = 0

def update_stats(img):

    global stats, n
    
    for y in range(480):
        for x in range(640):
            stats[y, x, img[y, x]] += 1
    
    n += 1


cap = cv2.VideoCapture(0)

if __name__ == "__main__":
    ret, bg = cap.read()
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        update_stats(gray)
        if n == N_FRAMES:

            bg = np.argmax(stats, axis=2)
            stats = np.zeros((480, 640, 256))
            n = 0


        cv2.imshow("frame", np.array(bg, np.uint8))

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
