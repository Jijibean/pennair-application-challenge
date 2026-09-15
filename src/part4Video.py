import os
import sys

import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detector import (build_mask_flood, find_shapes, find_circle,
                      compute_depth, back_project)

vid = cv2.VideoCapture('data/PennAirVideo2.mp4')
if not vid.isOpened():
    raise FileNotFoundError("check the video path")

while True:
    ret, frame = vid.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    principal = (w / 2, h / 2)

    mask = build_mask_flood(frame)
    shapes = find_shapes(mask, min_area=5000)

    circle = find_circle(shapes)
    Z = compute_depth(circle[0]) if circle else None

    for contour, (cx, cy) in shapes:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(frame, [approx], -1, (0, 0, 255), 2)   
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        if Z is not None:
            X, Y = back_project(cx, cy, Z, principal)
            label = f"({X:.0f}, {Y:.0f}, {Z:.0f})"
        else:
            label = f"({cx}, {cy})"

        cv2.putText(frame, label, (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('result', frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

vid.release()
cv2.destroyAllWindows()