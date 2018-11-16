import numpy as np
import cv2

camera_matrix = np.array( [[660.40356585, 0., 364.47045685],
 [0., 680.26168193, 147.68865612],
 [0., 0., 1.]])
dist_coefs = np.array([-1.06520912, 0.95815053, 0.01767912, -0.07019785, -1.40598956])

size = (640, 480)
cap = cv2.VideoCapture(0)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (size[0], size[1]), 1, (size[0], size[1]))
x, y, w, h = roi

while True:
    res, frame = cap.read()
    frame = cv2.undistort(frame, camera_matrix, dist_coefs, None, newcameramtx)
    frame = frame[y:y+h-50, x+70:x+w-20]

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()