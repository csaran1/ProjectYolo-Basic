import cv2
import numpy as np


video_path = r'C:\Users\csaran1\Documents\GitHub\YOLO-VD\Basic\Videos\cars.mp4'
cap = cv2.VideoCapture(video_path)
l=0
def extract_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=5)

    if lines is not None:
        # Calculating the average angle of detected lines
        angles = [np.arctan2((y2 - y1), (x2 - x1)) for x1, y1, x2, y2 in lines[:, 0]]
        average_angle = np.degrees(np.mean(angles))
        return average_angle
    else:
        return 0

lane_changes_values = []

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Extract relevant features from the frame (e.g., lane changes, speed, braking)
    average_angle = extract_features(frame)
    lane_changes_values.append(average_angle)

    # Highlight frames with detected lane changes (modify threshold as needed)
    if np.abs(average_angle) > 20:  # Adjust the threshold as needed
        #cv2.rectangle(frame, (100, 100), (200, 200), (0, 0, 255), 2)
        cv2.putText(frame, 'Lane Change Detected', (120, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        l+=1

    # Display the processed frame
    cv2.imshow('Lane Change Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
cap.release()
#cv2.destroyAllWindows()
if l>0:
    print("Lane change detected")


#Code is runnable and it is detecting lane change  if happens in video