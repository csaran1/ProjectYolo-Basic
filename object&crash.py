from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import datetime

# Object Detection Setup
cap = cv2.VideoCapture(r"https://trafficview.org/map/cctv-player")
model = YOLO(r"C:\Users\csaran1\Documents\GitHub\YOLO-VD\Yolo-weights\yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Crash Detection Setup
crash_model = MobileNetV2(weights='imagenet')
#video_capture = cv2.VideoCapture(r"C:\Users\csaran1\Documents\GitHub\YOLO-VD\Basic\Videos\Collision.mp4")
frame_number = 0
count =0
crash_timestamp = None
d=[]

#logging.basicConfig(filename='crash.log', level=logging.INFO)
while True:
    success, frame = cap.read()

    if not success:
        break

    # Object Detection
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Crash Detection
    frame_resized = cv2.resize(frame, (224, 224))
    predictions = crash_model.predict(np.expand_dims(frame_resized, axis=0))
    decoded_predictions = decode_predictions(predictions)
    top_prediction = decoded_predictions[0][0]
    start_time = datetime.datetime.strptime("00:00:00", "%H:%M:%S")

    if top_prediction[2] <= 0.6 and top_prediction[2] >= 0.4:  #top_prediction[2]<=0.1:
        print("Crash detected in this frame!")
        count+=1
        #frame_filename = f'crash_frame_{frame_number}.jpg'
        current_time = start_time + datetime.timedelta(seconds=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        d.append(current_time)
        #print(current_time)
        #frame_timestamp = video_capture.get(cv2.CAP_PROP_POS_MSEC)  # Get the timestamp of the current frame
        '''if crash_timestamp is None:
            crash_timestamp = frame_timestamp
            logging.info(f"Crash detected at : {frame_timestamp}")'''
        #cv2.imwrite(frame_filename, frame)
        frame_number += 1


    # Display the frame with detections
    cv2.imshow("Video Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if count>0:
    print("Crash happened in this video within duration of " , d[0],"to ", d[-1])
else:
    print("NO CRASH")



#Code is runnable and it detects the objects and also any crash happened in video or not but it does not gives you the cause of the detection