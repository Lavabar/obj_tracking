import numpy as np
import cv2

camera_matrix = np.array([[413.6867353, 0., 355.23033949],
 [0., 582.72597382, 233.73871734],
 [0., 0., 1.]])
dist_coefs = np.array([-2.90592148e-01, -9.64465709e-01, 1.12285534e-03, -3.83937338e-03, 2.81284524e+00])

size = (640, 480)
#cap = cv2.VideoCapture(0)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (size[0], size[1]), 1, (size[0], size[1]))
x, y, w, h = roi

# reading video
path = "/home/user/videos/271118_1100/106_(27-11-18_11\'00\'12).avi"
cap = cv2.VideoCapture(path)
while cap.isOpened():
    res, frame = cap.read()
    #frame = cv2.imread("test_frame.jpg")
    frame = cv2.undistort(frame, camera_matrix, dist_coefs, None, newcameramtx)
    frame = frame[y:y+h, x:x+w]
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()