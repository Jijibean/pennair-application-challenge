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

cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
