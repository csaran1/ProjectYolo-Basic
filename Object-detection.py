# The old code working
# It calculates distance between consecutive vehicle id's in a frame
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from Centroidtracker import CentroidTracker
import numpy as np

cap = cv2.VideoCapture(r"C:\Users\csaran1\Documents\GitHub\YOLO-VD\Basic\Videos\carsVideo.mp4")
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

prev_frame_time = 0
new_frame_time = 0
tracker = CentroidTracker(maxDisappeared=1, maxDistance=80)

def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

def calculate_distance(pixel_width, known_width, focal_length):
    return (known_width * focal_length) / pixel_width

# Assuming known width of a car in meters
known_car_width_inches = 70.0
conversion_factor= 0.05
# Assuming focal length in pixels (you may need to calibrate this value)
#focal_length = 25489.655172413793

total_distance=0
num_distances=0
while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    rects = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            rects.append((x1, y1, x2, y2))
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    rects = np.array(rects)
    rects_after_nms = non_max_suppression_fast(rects, overlapThresh=0.5)
    tracked_bboxes = tracker.update(rects_after_nms)
    tracked_objects = {}
    tracked_ids = list(tracked_bboxes.keys())
    for object_id, bbox in tracked_bboxes.items():
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.putText(img, f'ID: {object_id}', (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    for i in range(len(tracked_ids) - 1):
        current_id = tracked_ids[i]
        next_id = tracked_ids[i + 1]

        (x1, y1, x2, y2), (x3, y3, x4, y4) = tracked_bboxes[current_id], tracked_bboxes[next_id]

        # Calculate Euclidean distance between centroids
        current_centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
        next_centroid = ((x3 + x4) // 2, (y3 + y4) // 2)
        distance_pixels = math.sqrt(
            (next_centroid[0] - current_centroid[0]) ** 2 + (next_centroid[1] - current_centroid[1]) ** 2)

        # Convert distance from pixels to inches
        distance_inches = distance_pixels * conversion_factor


        print(f"Distance between vehicle {current_id} and vehicle {next_id}: {distance_inches:.2f} inches")
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
