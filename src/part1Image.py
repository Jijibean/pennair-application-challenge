import cv2

img = cv2.imread('data/PennAirImageStatic.png')
if img is None:
    raise FileNotFoundError("check the path")

blurImg = cv2.GaussianBlur(img, (5, 5), 0)
hsv = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

mask = cv2.inRange(v, 130, 255)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    area = cv2.contourArea(contour)
    if area > 3300:
        cv2.drawContours(img, [contour], -1, (0, 0,255), 2)
        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(img, f"({cx}, {cy})", (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow('mask', mask)
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
