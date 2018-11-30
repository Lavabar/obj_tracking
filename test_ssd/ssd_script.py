# import the necessary packages
import numpy as np
import imutils
import cv2

path_prototxt = "/home/user/projects/obj_tracking/test_ssd/deploy.prototxt"
path_model = "/home/user/projects/obj_tracking/test_ssd/MobileNetSSD_deploy.caffemodel"
threshold = 0.5

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(path_prototxt, path_model)

import os

path = "/home/user/projects/obj_tracking/test_videos/Crowd_PETS09/S1/L1/Time_13-59/View_001/"
ls = os.listdir(path)
ls.sort()

for fr_name in ls:
	img = cv2.imread(path+fr_name)

	img = imutils.resize(img, width=500)

	(H, W) = img.shape[:2]
	#convert the frame to a blob and pass the blob through the
	# network and obtain the detections
	blob = cv2.dnn.blobFromImage(img, 0.007843, (W, H), 127.5)
	net.setInput(blob)
	detections = net.forward()
	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated
		# with the prediction
		confidence = detections[0, 0, i, 2]
		# filter out weak detections by requiring a minimum
		# confidence
		if confidence > threshold:
			# extract the index of the class label from the
			# detections list
			idx = int(detections[0, 0, i, 1])
			# if the class label is not a person, ignore it
			if CLASSES[idx] != "person":
				continue
				
			# compute the (x, y)-coordinates of the bounding box
			# for the object
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")
			
			cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 1)
	cv2.imshow('frame', img)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cv2.destroyAllWindows()