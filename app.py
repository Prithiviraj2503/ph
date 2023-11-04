import cv2 as cv
import numpy as np

# Load the YOLOv7 Tiny model
net1 = cv.dnn.readNet('yolov7_tiny.weights', 'yolov7_tiny.cfg')
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

def findPotholes(frame):
    classes, scores, boxes = model1.detect(frame, confThreshold=0.5, nmsThreshold=0.4)

    pothole_coordinates = []

    if classes is not None:
        for (class_id, score, box) in zip(classes, scores, boxes):
            if class_id == 0:  # Assuming class_id 0 corresponds to potholes
                x, y, w, h = box
                pothole_coordinates.append((x, y, w, h))

    return pothole_coordinates

cap = cv.VideoCapture(0)  # Use the default camera (0) for webcam feed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pothole_coordinates = findPotholes(frame)

    # Draw rectangles around detected potholes
    for (x, y, w, h) in pothole_coordinates:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('Pothole Detection', frame)
    key = cv.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
