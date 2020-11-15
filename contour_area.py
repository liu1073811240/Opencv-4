import cv2

'''面积，周长，重心'''
# gray = cv2.imread("./images/21.jpg", 0)
gray = cv2.imread("./images/22.jpg", 0)

ret, binary = cv2.threshold(gray, 127, 255, 0)

contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 重心
# moments(array, binaryImage=None)
print(contours[0])  # 轮廓点的坐标
M = cv2.moments(contours[0])  # 矩
print(M)

cx = int(M['m10']) / M['m00']
cy = int(M['m01']) / M['m00']
print("重心：", cx, cy)

# 面积
# contourArea(contour, oriented=None)
area = cv2.contourArea(contours[0])
print("面积：", area)

# 周长
# arcLength(curve, closed)
perimeter = cv2.arcLength(contours[0], True)
print("周长：", perimeter)

cv2.imshow("gray", gray)
cv2.imshow("binary", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
