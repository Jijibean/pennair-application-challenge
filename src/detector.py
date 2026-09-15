import cv2
import numpy as np

FX = 2564.3186869
FY = 2569.70273111
CIRCLE_RADIUS_IN = 10.0


def circularity(contour):
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    if perim == 0:
        return 0
    return 4 * np.pi * area / (perim * perim)


def find_circle(shapes):
    best = max(shapes, key=lambda s: circularity(s[0]))
    return best if circularity(best[0]) > 0.85 else None

def compute_depth(circle_contour):
    (_, _), radius_px = cv2.minEnclosingCircle(circle_contour)
    if radius_px == 0:
        return None
    return FX * CIRCLE_RADIUS_IN / radius_px


def back_project(cx, cy, Z, principal):
    px, py = principal
    X = (cx - px) * Z / FX
    Y = (cy - py) * Z / FY
    return X, Y


def build_mask(img):
    blurImg = cv2.GaussianBlur(img, (7, 7), 0)
    hsv = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    mask = cv2.inRange(v, 150, 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def find_shapes(mask, min_area=1000, max_area=1e9, min_solidity=0.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area == 0 or area / hull_area < min_solidity:
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

    result = cv2.bitwise_not(filled)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    return cv2.morphologyEx(result, cv2.MORPH_OPEN, open_kernel)