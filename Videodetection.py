#Good code for distance without direction
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from Centroidtracker import CentroidTracker
import numpy as np


# Function to calculate direction based on optical flow vectors
def calculate_direction(flow):
    mean_flow = np.mean(flow, axis=(0, 1))
    dx, dy = mean_flow

    # Determine direction based on sign of motion
    if dx < 0:
        return "Vehicle moving towards camera"
    elif dx > 0:
        return "Vehicle moving away from camera"
    else:
        return "Vehicle not moving horizontally"


# Initialize video capture
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

#This function performs non-maximum suppression on a set of bounding boxes to remove overlapping boxes, keeping only the ones with the highest confidence.
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

# Function to calculate distance using focal length
def calculate_distance(pixel_width, known_width, focal_length):
    return (known_width * focal_length) / pixel_width

# Focal length parameters
measured_distance = 11088  # distance between traffic video and car in inches approximately
real_width = 69.6  # the average width of car in inches
width_in_rf_image = 160.0  # width of the object in the reference image (in pixels)
focal_length = (width_in_rf_image * measured_distance) / real_width

print("Focal length:", focal_length)

# Function to calculate angle from centroid
def calculate_angle_from_centroid(centroid, reference_centroid):
    if centroid is not None and reference_centroid is not None:
        x1, y1 = centroid
        x2, y2 = reference_centroid
        #print(centroid)
        #print(reference_centroid)
        # Calculate the vector from the reference centroid to the current centroid
        delta_x = x1 - x2
        delta_y = y1 - y2
        # Calculate the angle of the vector
        angle_rad = math.atan2(delta_y, delta_x)
        #print(angle_rad)
        angle_deg = math.degrees(angle_rad)
        #print(angle_deg)
        print("Angle from reference centroid to current centroid:", angle_deg, "(delta_y:", delta_y, ", delta_x:",delta_x, ")")
        return angle_rad, angle_deg
    else:
        return None

# Main loop to process video frames
reference_vehicle_id = 0
followed_vehicles = []
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    rects = []
    # Calculate optical flow between frames
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    direction = calculate_direction(flow)
    print("Direction of vehicle movement:", direction)
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
            # cv2.circle(img, (cx, cy), 4, (255,0,0), -1)
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    rects = np.array(rects)
    rects_after_nms = non_max_suppression_fast(rects, overlapThresh=0.5)
    tracked_bboxes = tracker.update(rects_after_nms)
    tracked_objects = {}
    tracked_ids = list(tracked_bboxes.keys())
    print("Tracked ids list", tracked_ids)
    if reference_vehicle_id not in tracked_ids:
        print(followed_vehicles)
        for vehicle_id in followed_vehicles:
            if vehicle_id in tracked_ids:
                next_vehicle_index = followed_vehicles.index(vehicle_id)
                print("followed vehicles list:", followed_vehicles)
                break
        if next_vehicle_index < len(followed_vehicles):
            reference_vehicle_id = followed_vehicles[next_vehicle_index]
            print("Reference Vehicle ID:", reference_vehicle_id)
    # Iterate over tracked objects
    for object_id, bbox in tracked_bboxes.items():
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.putText(img, f'ID: {object_id}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if object_id == reference_vehicle_id:
            print(f"Reference Vehicle ID {reference_vehicle_id}: Centroid: ({cx}, {cy})")
            reference_centroid = (cx, cy)

    # Iterate over other tracked vehicles
    for vehicle_id, bbox in tracked_bboxes.items():
        if vehicle_id == reference_vehicle_id:
            continue

        current_centroid = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        #print(f"Vehicle ID: {vehicle_id}, Centroid: {current_centroid}")
        angle_r, angle_d = calculate_angle_from_centroid(current_centroid, reference_centroid)
        if abs(angle_r) < 2 and  abs(angle_d) <110:
            # Get the class name of the current vehicle
            # cls_name = classNames[int(results[vehicle_id].cls[0])]
            if vehicle_id not in followed_vehicles:
                followed_vehicles.append(vehicle_id)
                print(followed_vehicles)
            # Calculate distance between centroids
            distance_pixels = math.sqrt((current_centroid[0] - reference_centroid[0]) ** 2 +
                                        (current_centroid[1] - reference_centroid[1]) ** 2)
            # Calculate distance in inches
            known_car_width_inches = 70.0  # Set the known width of the car in inches
            distance_inches = calculate_distance(distance_pixels, known_car_width_inches, focal_length)
            print(f"Distance between vehicle {reference_vehicle_id} and vehicle {vehicle_id}: {distance_inches:.2f} inches")

        # Visualize centroid on the image
        cv2.circle(img, current_centroid, 4, (0, 255, 0), -1)  # green
        cv2.putText(img, f'ID: {vehicle_id}', current_centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame with tracking information
    cv2.imshow("Frame", img)

    # Press 'Esc' to exit
    if cv2.waitKey(0) & 0xFF == 27:
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
