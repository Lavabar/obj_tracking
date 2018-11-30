import cv2
path = "/home/user/videos/271118_1100/106_(27-11-18_11\'00\'12).avi"
cap = cv2.VideoCapture(path)

# waiting a few moments while camera focusing
while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("test_frame.jpg", frame)
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()