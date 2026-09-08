import os
import sys

import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detector import build_mask_flood, find_shapes

vid = cv2.VideoCapture('data/PennAirVideo2.mp4')
if not vid.isOpened():
    raise FileNotFoundError("check the video path")

while True:
    ret, frame = vid.read()
    if not ret:
        break

    mask = build_mask_flood(frame)
    shapes = find_shapes(mask, min_area=5000)

    for contour, (cx, cy) in shapes:
        cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"({cx}, {cy})", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('mask', mask)
    cv2.imshow('result', frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

vid.release()
cv2.destroyAllWindows()