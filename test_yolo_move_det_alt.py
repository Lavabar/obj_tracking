import numpy as np
import cv2
from skimage.measure import compare_ssim
import imutils

from darkflow.net.build import TFNet

options = {"model": "yolov2-tiny.cfg", "load": "yolov2-tiny.weights", "threshold": 0.1, "labels": "labels.txt"}

tfnet = TFNet(options)

def get_img_objs(original, frame):
    res = frame.copy()
    
    grayA = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    (score, diff) = compare_ssim(grayA, grayB, full=True)
    if score > 0.90:
        return res
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    result = tfnet.return_predict(frame)

    for obj in result:
        if obj['label'] == 'person' and obj['confidence'] >= 0.7:
            x1 = obj['topleft']['x']
            y1 = obj['topleft']['y']
            x2 = obj['bottomright']['x']
            y2 = obj['bottomright']['y']
            
            cut = thresh[y1:y2, x1:x2]
            w = x2 - x1
            h = y2 - y1
            pts = np.count_nonzero(cut)
            if pts > w * h * 0.4:
                cv2.rectangle(res, (x1, y1), (x2, y2), (0, 0, 255), 3)
                # + start tracking object if it is not tracked
                # + give an id
                # + find legs
                # + put moves on the map

    return res

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    ret, original = cap.read()
    for _ in range(50):
        ret, original = cap.read()

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_objs = get_img_objs(original, frame)

        # Display the resulting frame
        cv2.imshow('frame', img_objs)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()