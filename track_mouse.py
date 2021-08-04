import cv2
import numpy as np
from mouse_detection.tracker import *
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("resources/mitmouse.mp4")
# object detector:
object_detector = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=24)
# tracking
tracker = EuclideanDistTracker()

# morphology filters
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel10 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
# forgeting factor
alpha = 0.1
mask_hist = None
eps = 1E-12
xx = []
last_coords = 4 * [np.nan]
while True:
    try:
        ret, frame = cap.read()
        # object detection
        mask = object_detector.apply(frame)
        if mask_hist is None:
            mask_hist = np.zeros((frame.shape[0], frame.shape[1]), dtype=float)
        _, mask = cv2.threshold(mask, 254,255,cv2.THRESH_BINARY)

        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel10)

        mask_hist = mask.astype(float) + alpha * mask_hist
        mask = mask_hist
        # print(mask)
        mask = 255*(mask - mask.min())/(eps+mask.max()-mask.min())
        mask = mask.astype(frame.dtype)


        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            frame_detections = []
            if area > 400:
                # cv2.drawContours(frame, [cnt], -1, (0,255,0),2)
                x, y, h, w = cv2.boundingRect(cnt)
                # cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
                frame_detections.append([x, y, h, w])
            if len(frame_detections)>0:
                print('frame detections: ', frame_detections)
                detections_avg = np.mean(frame_detections, axis=0)
                print('detection avg: ', detections_avg)
                detections.append(detections_avg.astype(int).tolist())
                # xx.append(detections_avg[0])
            # else:
            #     xx.append(np.nan)
        # detection
        boxes_id = tracker.update(detections)
        # 2. Object Tracking
        boxes_ids = tracker.update(detections)
        frame_coords = []
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            frame_coords.append([x, y, w, h])
            cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        if len(frame_coords)>0:
            last_coords = np.mean(frame_coords, axis=0)
            xx.append(last_coords)
        else:
            xx.append(last_coords)
        # plotting a cyan squere on the
        if not np.isnan(np.sum(last_coords)):
            x, y, w, h = last_coords.astype(int)
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            W = 100
            H = 100
            cv2.rectangle(frame, (cx-W//2, cy-H//2), (cx + W//2, cy + H//2), (255,255,0), 3)

        cv2.imshow("Frame", frame)
        # cv2.imshow("Mask", mask)
        # cv2.imshow("contours", frame_cnt)
        key = cv2.waitKey(30)
        if key == 27:
            break
    except Exception as e:
        print(e)
        plt.plot(xx)
        plt.show()
        break

cap.release()
cv2.destroyAllWindows()
