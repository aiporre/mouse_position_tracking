import cv2


cap = cv2.VideoCapture("resources/mitmouse.mp4")
# object detector:
object_detector = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=16)
# morphology filters
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

while True:
    ret, frame = cap.read()
    # object detection
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254,255,cv2.THRESH_BINARY)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            # cv2.drawContours(frame, [cnt], -1, (0,255,0),2)
            x, y, h, w = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
    # cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    # cv2.imshow("contours", frame_cnt)
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
