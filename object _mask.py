import cv2
import numpy as np

# 对象掩码mask
img = cv2.imread("./images/23.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, 0 | 8)

contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros(img.shape, np.uint8)
cv2.drawContours(mask, contours, -1, (255, 0, 0), -1)

pixel_points = np.transpose(np.nonzero(mask))
print(pixel_points)

cv2.imshow("img", img)
cv2.imshow("mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()



