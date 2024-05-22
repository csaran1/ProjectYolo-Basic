import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from Centroidtracker import *
import time
from math import sqrt
model = YOLO(r"C:\Users\csaran1\Documents\GitHub\YOLO-VD\Yolo-weights\yolov8n.pt")


def dist(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture(r"C:\Users\csaran1\PycharmProjects\pythonProject-Yolo\yolov8counting-trackingvehicles\veh2\cars.mp4")

class_list=["Car","Truck","Bus","Person"]
# print(class_list)

count = 0

tracker2 = Tracker()

cy1 = 250
cy2 = 300

offset = 10

vh_down = {}
counter = []

vh_up = {}
counter1 = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    #   print(results)
    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")
    #    print(px)
    list = []

    for index, row in px.iterrows():
        #        print(row)

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        #d = int(row[5])
        class_list= ["car"]
        #c = class_list[d]
        if 'car' in class_list:
            list.append([x1, y1, x2, y2])
    bbox_id = tracker2.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        cv2.circle(frame, (cx,cy), 3, (0, 255, 0), -1)
        cv2.putText(frame,(str(id)), (cx,cy), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
        # print(f"cy1: {cy1}, cy: {cy}, offset: {offset}")
        if cy1 < (cy + offset) and cy1 > (cy - offset):  # cy is centroid of vehicle
            print("In 1st loop")
            vh_down[id] = time.time()
            print(vh_down)
        if id in vh_down:
            count+=1
            print("In 2nd condition loop")
            print(f"cy2: {cy2}, cy: {cy}, offset: {offset}")
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                print("calculating")
                elapsed_time = time.time() - vh_down[id]  # gives time taken for vehicle to come from l2 to l1
                print(elapsed_time)
                if counter.count(id) == 0:
                    counter.append(id)
                    print("Inside loop")
                    distance = 10  # meters #distance between ines
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    print(a_speed_kh)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 255, 255), 2)

        #####going UP#####
        if cy2 < (cy + offset) and cy2 > (cy - offset):
            vh_up[id] = time.time()
        if id in vh_up:

            if cy1 < (cy + offset) and cy1 > (cy - offset):
                elapsed1_time = time.time() - vh_up[id]

                if counter1.count(id) == 0:
                    counter1.append(id)
                    distance1 = 10  # meters
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    print(a_speed_kh1)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 255, 255), 2)

    cv2.line(frame, (238, cy1-offset), (478, cy1-offset), (255, 255, 255), 1)  # 238,478
    # cv2.line(frame, (238, cy1), (478, cy1), (255, 255, 255), 1)  # 238,478
    cv2.line(frame, (238, cy1+10), (478, cy1+10), (255, 255, 255), 1)  # 238,478

    cv2.putText(frame, ('L1'), (238, 261), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)  # 478,288)(238,261),

    cv2.line(frame, (121, cy2-offset), (473, cy2-offset), (255, 255, 255), 1)  # 121,473
    # cv2.line(frame, (121, cy2), (473, cy2), (255, 255, 255), 1)  # 121,473
    cv2.line(frame, (121, cy2+offset), (473, cy2+offset), (255, 255, 255), 1)  # 121,473


    cv2.putText(frame, ('L2'), (121, 321), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)  # 473,324#(121,321)
    d = (len(counter))
    u = (len(counter1))
    cv2.putText(frame, ('goingdown:-') + str(d), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, ('goingup:-') + str(u), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
