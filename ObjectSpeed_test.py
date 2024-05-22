from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Object Detection Setup
cap = cv2.VideoCapture(r"C:\Users\csaran1\Documents\GitHub\YOLO-VD\Basic\Videos\cars.mp4")
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

# Speed Calculation Setup
line_start = (0, 450)
line_end = (600, 450)
pixel_to_feet = 1 / (line_end[0] - line_start[0])
pixel_to_mile = pixel_to_feet / 5280
vehicle_crossed_time = {}
min_speed_threshold = 5.0

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)

last_center_x=0
last_y1=0
while True:
    success, frame = cap.read()

    if not success:
        break

    # Draw line for speed calculation
    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)

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

            # Speed Calculation
            center_x = x1 + w / 2
            vehicle_id = id(box)

            if line_start[0] < center_x < line_end[0]:
                print("Vehicle crossed the line")

                if vehicle_id not in vehicle_crossed_time:
                    vehicle_crossed_time[vehicle_id] = time.time()
                else:
                    if vehicle_id in vehicle_crossed_time:
                        time_difference = time.time() - vehicle_crossed_time[vehicle_id]
                        print("Time Difference:", time_difference, "seconds")

                        if time_difference > 0:
                            # Calculate distance traveled using Euclidean distance
                            distance_pixels = math.sqrt((center_x - last_center_x) ** 2 + (y1 - last_y1) ** 2)
                            distance_miles = distance_pixels * pixel_to_mile

                            # Calculate speed
                            speed_miles_per_hour = distance_miles / time_difference * 3600
                            print(f"Vehicle ID: {vehicle_id}, Speed (mph): {speed_miles_per_hour:.2f}")

                            if speed_miles_per_hour >= min_speed_threshold:
                                print("Speed above threshold!")

                            speed_text = f"Speed: {speed_miles_per_hour:.2f} mph"
                            cv2.putText(frame, speed_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 255, 255), 1, cv2.LINE_AA)

                        # Update the last known position for the next iteration
                        last_center_x = center_x
                        last_y1 = y1

                        vehicle_crossed_time.pop(vehicle_id, None)


    # Display the frame with detections
    cv2.imshow("Video Frame", frame)
    current_fps = cap.get(cv2.CAP_PROP_FPS)
    print("Current FPS:", current_fps)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#Code is runnable ang getting the output with speeddisplaying to each vehicle but the numbers are not convincible
# Need to analyze the code more thouroughly and understand the output on how it is coming ang y it is coming like this