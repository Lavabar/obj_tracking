import cv2
from darkflow.net.build import TFNet

options = {"model": "yolov2-tiny.cfg", "load": "yolov2-tiny.weights", "threshold": 0.1, "labels": "labels.txt"}

tfnet = TFNet(options)

cap = cv2.VideoCapture(0)
counter = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    counter += 1
    #if counter == 10:
    result = tfnet.return_predict(frame)
    #    print(result)

    for obj in result:
        if obj['label'] == 'person' and obj['confidence'] >= 0.6:
            x1 = obj['topleft']['x']
            y1 = obj['topleft']['y']
            x2 = obj['bottomright']['x']
            y2 = obj['bottomright']['y']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    counter = 0
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()