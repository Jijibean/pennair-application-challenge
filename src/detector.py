import cv2


def build_mask(img):
    blurImg = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    mask = cv2.inRange(v, 130, 255)

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