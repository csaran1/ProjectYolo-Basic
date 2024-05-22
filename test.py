from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from Centroidtracker import CentroidTracker
import numpy as np

# Load YOLO
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

def detect_objects(frame):
    # Detecting objects
    results = model(frame, stream=True)
    rects = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            rects.append((x1, y1, x2, y2))
            cvzone.cornerRect(frame, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(frame, (cx, cy), 4, (255,0,0), -1)
            cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    return rects

def analyze_traffic(objects):
    # Assuming the lane divisions are known
    lane_divisions = [50, 150, 250]  # Example lane divisions, adjust according to your video

    # Initialize dictionary to store lane information
    lane_info = {}

    for obj in objects:
        x, y, w, h = obj
        lane_found = False
        for i, division in enumerate(lane_divisions):
            if x < division:
                lane_info[i + 1] = obj  # Assuming lane numbering starts from 1
                lane_found = True
                break
        if not lane_found:
            lane_info[len(lane_divisions) + 1] = obj  # Object is beyond defined lanes

    return lane_info

# Function to process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the frame
        objects = detect_objects(frame)
        print(objects)
        # Analyze traffic and get leading vehicles
        leading_vehicles = analyze_traffic(objects)

        print("Leading vehicles in each lane:")
        for lane, vehicle in leading_vehicles.items():
            print("Lane:", lane, "Vehicle:", vehicle)

        # Display the frame with bounding boxes
        for obj in objects:
            x, y, w, h = obj
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
process_video(r"C:\Users\csaran1\Documents\GitHub\YOLO-VD\Basic\Videos\carsVideo.mp4")
