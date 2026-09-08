import cv2
import numpy as np


def build_mask(img):
    blurImg = cv2.GaussianBlur(img, (7, 7), 0)
    hsv = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    mask = cv2.inRange(v, 150, 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def find_shapes(mask, min_area=1000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        shapes.append((contour, (cx, cy)))
    return shapes

def build_mask_flood(img, var_thresh=100, window=11):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY).astype("float32")
    mean = cv2.blur(gray, (window, window))
    sq_mean = cv2.blur(gray * gray, (window, window))
    variance = sq_mean - mean * mean

    bg = cv2.inRange(variance, var_thresh, 1e9)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, kernel)

    h, w = bg.shape
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    filled = bg.copy()
    cv2.floodFill(filled, ff_mask, (0, 0), 255)

    return cv2.bitwise_not(filled)