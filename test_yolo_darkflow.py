import cv2
from darkflow.net.build import TFNet

options = {"model": "yolov2-tiny.cfg", "load": "yolov2-tiny.weights", "threshold": 0.1, "labels": "labels.txt"}

tfnet = TFNet(options)

imgcv = cv2.imread("people.jpg")
result = tfnet.return_predict(imgcv)
print(result)

for obj in result:
    if obj['label'] == 'person' and obj['confidence'] >= 0.7:
        x1 = obj['topleft']['x']
        y1 = obj['topleft']['y']
        x2 = obj['bottomright']['x']
        y2 = obj['bottomright']['y']
        cv2.rectangle(imgcv, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('frame', imgcv)
key = False
while not key:
    key = cv2.waitKey(1) & 0xFF == ord('q')