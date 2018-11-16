import numpy as np
import cv2

camera_matrix = np.array([[1.26125746e+03, 0.00000000e+00, 9.40592038e+02],
                          [0.00000000e+00, 1.21705719e+03, 5.96848905e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coefs = np.array([-3.18345478e+01, 7.26874187e+02, -1.20480816e-01, 9.43789095e-02, 5.28916586e-01])


cap = cv2.VideoCapture(0)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (size[0], size[1]), 1, (size[0], size[1]))
x, y, w, h = roi
M = cv2.getRotationMatrix2D((size[0]/2,size[1]/2),5,1)

while True:
    frame = cap.read()
    frame = cv2.undistort(frame, camera_matrix, dist_coefs, None, newcameramtx)
    frame = frame[y:y+h-50, x+70:x+w-20]

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()