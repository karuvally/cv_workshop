#!/usr/bin/python2

# import the essential stuff
import numpy as np
import cv2
import matplotlib.pyplot as plt
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time

# declare essential variables
prototxt_file = "downloads/MobileNetSSD_deploy.prototxt.txt"
model = "downloads/MobileNetSSD_deploy.caffemodel"
required_confidence = 0.2


# define the labels supported by the model
supported_labels = ["background", "aeroplane", "bicycle", "bird", "boat",
                    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                    "sofa", "train", "tvmonitor"]


# load the model
neural_net = cv2.dnn.readNetFromCaffe(prototxt_file, model)


# initialize video stream
video_stream = VideoStream(src="downloads/sample_video.avi").start()
time.sleep(2.0)
fps = FPS().start()


# loop over frames
while True:
    frame = video_stream.read()
    frame = imutils.resize(frame, width=400)
    
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    
    neural_net.setInput(blob)
    detections = neural_net.forward()
    
    # show the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
    
        # collect info about detected objects
        if confidence > required_confidence:
            object_index = int(detections[0, 0, i, 1])
            bounding_box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, box_width, box_height) = bounding_box.astype("int")
        
            # the labelling stuff
            label = "{}: {:.2f}%".format(supported_labels[object_index], confidence*100)
            cv2.rectangle(frame, (x, y), (box_width, box_height), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # show the image
    cv2.imshow("Frame", frame)
    
    # set up the exit key
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
        
    # update fps counter
    fps.update()

# if stopped, show FPS info
fps.stop()
print("FPS: ", fps.fps())

# kill the video stream
video_stream.stop()
cv2.destroyAllWindows()

