from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture(r"C:\Users\csaran1\Documents\YOLO\Object-Detection-101\Videos\motorbikes.mp4")  # For Video

model = YOLO(r"../Yolo-Weights/yolov8n.pt")

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

# Define crash detection parameters
crash_objects = ["car", "motorbike"]  # Objects relevant to crashes

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    crash_detected = False  # Flag to indicate a crash

    # Create lists to store the coordinates of cars and motorbikes
    car_coords = []
    motorbike_coords = []
    count=0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            detected_class = classNames[cls]

            # Draw bounding boxes and labels
            cvzone.cornerRect(img, (x1, y1, w, h))
            cvzone.putTextRect(img, f'{detected_class} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            # Check for a crash (collision)
            if detected_class in crash_objects:
                crash_detected = True
                count+=1

    if crash_detected:
        # Perform actions when a crash is detected, e.g., save a snapshot or trigger an alert
        cv2.putText(img, "CRASH", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #print("Crash detected!")
    if count>1:
        print("Crash deected")


    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('b'):
        break

#Code is runnable but it is showing Crash happened even if there is no crash in video