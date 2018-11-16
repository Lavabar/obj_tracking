import numpy as np
import cv2
from skimage.measure import compare_ssim
import imutils


def get_img_objs(frame):
    res = frame.copy()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    heads = head_class.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(20, 20))
    for (x, y, w, h) in heads:
        cv2.rectangle(res, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return res

head_class = cv2.CascadeClassifier("models/haar/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_objs = get_img_objs(frame)

    # Display the resulting frame
    cv2.imshow('frame', img_objs)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()