import numpy as np
import cv2
from skimage.measure import compare_ssim
import imutils

from darkflow.net.build import TFNet


class DCTP:
    """
        DCTP stands for Detection Classifying Tracking Positioning
    """

    def __init__(self):
        # parameters for yolo
        path = "models/yolo/"
        self.options = {"model": path+"yolov2-tiny.cfg", "load": path+"yolov2-tiny.weights", "threshold": 0.1, "labels": path+"labels.txt"}
        # init darkflow yolo implementation
        self.tfnet = TFNet(self.options)
        self.head_class = cv2.CascadeClassifier("models/haar/haarcascade_frontalface_default.xml")

    def get_objs(self, original, frame):
        """
            looking for persons in areas where actions were detected
            returning list of tuples (x, y, w, h)
        """
        res = []
        # two grayscale images for move-detection
        grayA = original
        grayB = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # scoring difference of current frame and original frame
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        if score > 0.90:
            return res
        # getting binarized image with moving areas only
        diff = (diff * 255).astype("uint8")
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # now looking for objects
        result = self.tfnet.return_predict(frame)

        for obj in result:
            if obj['label'] == 'person' and obj['confidence'] >= 0.7:
                x1 = obj['topleft']['x']
                y1 = obj['topleft']['y']
                x2 = obj['bottomright']['x']
                y2 = obj['bottomright']['y']
                # checking if found object wasnt moving
                cut = thresh[y1:y2, x1:x2]
                w = x2 - x1
                h = y2 - y1
                pts = np.count_nonzero(cut)
                if pts > w * h * 0.4:
                    res.append((x1, y1, x2-x1, y2-y1))
        return res

    def get_haar_objs(self, original, frame):
        res = frame.copy()
    
        grayA = original
        grayB = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # scoring difference of current frame and original frame
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        if score > 0.90:
            return res
        # getting binarized image with moving areas only
        diff = (diff * 255).astype("uint8")
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # now looking for objects
        heads = self.head_class.detectMultiScale(grayB, scaleFactor=1.1, minNeighbors=0, minSize=(32, 64))
        for (x, y, w, h) in heads:
            cut = thresh[y:y+h, x:x+w]
            pts = np.count_nonzero(cut)
            if pts > w * h * 0.4:
                cv2.rectangle(res, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return res

    def get_img_objs(self, original, frame):
        res = frame.copy()
        objs = []
        objs = self.get_objs(original, frame)
        
        for obj in objs:
            (x, y, w, h) = [int(v) for v in obj]
            cv2.rectangle(res, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        return res

if __name__ == "__main__":
    dctp = DCTP()
    
    cap = cv2.VideoCapture(0)

    # waiting a few moments while camera focusing
    ret, original = cap.read()
    for _ in range(50):
        ret, original = cap.read()
    
    original = imutils.resize(original, width=600)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        frame = imutils.resize(frame, width=600)
        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_objs = dctp.get_img_objs(original, frame)

        # Display the resulting frame
        cv2.imshow('frame', img_objs)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()